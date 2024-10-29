import os
import shutil
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from essemble_model import EnsembleModel
from model_with_attention import GaitModelAttention
from model import GaitModel
from dataprep import load_data
from sklearn.metrics import mean_squared_error
import numpy as np
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate the EnsembleModel.')
    parser.add_argument('--img_dir', type=str, default='Data/Image', help='Directory for images')
    parser.add_argument('--legs_dir', type=str, default='Data/Legs', help='Directory for legs images')
    parser.add_argument('--csv_path', type=str, default='Data/filtered_data.csv', help='Path to CSV file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save models and results')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate for optimizer')
    return parser.parse_args()

def clear_output_dir(target_dir):
    """Clears the target directory before training."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

def rmse_loss(outputs, labels):
    """Calculate RMSE loss."""
    return torch.sqrt(nn.MSELoss()(outputs, labels))

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, target_dir, scheduler=None,
                early_stopping_patience=None):
    best_train_loss = float('inf')
    best_val_loss = float('inf')  # Initialize best_val_loss
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for (images1, images2, labels) in train_loader:
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images1, images2)  # Changed to EnsembleModel call
            loss = criterion(outputs.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (images1, images2, labels) in val_loader:
                images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
                outputs = model(images1, images2)
                loss = criterion(outputs.squeeze(-1), labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Step the scheduler
        if scheduler:
            scheduler.step(val_loss)

        # Get current learning rate
        lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {lr:.6f}')

        # Save the model based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(target_dir, 'best_model.pth'))
            print("Best model saved based on training loss...")

        # Early stopping logic
        if early_stopping_patience:
            if val_loss < best_val_loss - 0.01:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(target_dir, 'best_model_val.pth'))
                print("Best model saved based on validation loss...")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Training finished. Best training loss: {best_train_loss:.4f}")

def test_model(model, test_loader, criterion, device, target_dir):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for (images1, images2, labels) in test_loader:
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            outputs = model(images1, images2)
            loss = rmse_loss(outputs.squeeze(-1), labels)
            test_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    mae = np.mean(np.abs(all_labels - all_outputs))
    output_range = np.max(all_labels) - np.min(all_labels)
    normalized_mae = mae / output_range

    predictions_csv_path = os.path.join(target_dir, 'predictions.csv')
    with open(predictions_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['True Label', 'Predicted Label'])
        for true_label, predicted_label in zip(all_labels, all_outputs):
            writer.writerow([true_label, predicted_label])

    results_csv_path = os.path.join(target_dir, 'test_results.csv')
    with open(results_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Test Loss', 'Test MAE', 'Target Range', 'MAE/Range Ratio'])
        writer.writerow([avg_test_loss, mae, output_range, normalized_mae])

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Target Range: {output_range:.4f}")
    print(f"MAE/Range Ratio: {normalized_mae:.4f}")


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    targets = ['KneeFlex_maxExtension', 'cadence', 'steplen', 'GDI']

    for target in targets:
        for side in ['L', 'R']:
            print(f"\nTraining and evaluating for target: {target} on side: {side}")

            target_dir = os.path.join(args.output_dir, f"{target}_{side}")
            clear_output_dir(target_dir)

            log_path = os.path.join(target_dir, 'training_log.txt')
            with open(log_path, 'w') as log_file:
                log_file.write(f"Training log for {target} on {side} side\n")

            leg_dir = os.path.join(args.legs_dir, f'{side}_Images')  # Adjusted to point to the correct leg directory

            # Load data
            train_loader, val_loader, test_loader = load_data(args.csv_path, args.img_dir, leg_dir, batch_size=args.batch_size, transform=transform, target=target, side=side)

            model = GaitModel().to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

            train_model(model, train_loader, val_loader, args.epochs, criterion, optimizer, device, target_dir,
                        scheduler=scheduler, early_stopping_patience=10)

            best_model_path = os.path.join(target_dir, 'best_model_val.pth')
            model.load_state_dict(torch.load(best_model_path))

            print(f"\nTesting the best model for {target} on side: {side}...")
            test_model(model, test_loader, criterion, device, target_dir)

if __name__ == '__main__':
    main()
