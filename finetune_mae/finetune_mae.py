import torch
from data import NextImageBatchGenerator
from transformers import ViTImageProcessor, ViTMAEForPreTraining, ViTMAEConfig
from tqdm import tqdm

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    data_gen = NextImageBatchGenerator()

    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        with tqdm(enumerate(data_gen())) as pbar:
            for i, batch in pbar:
                pbar.set_description_str(f"Epoch {epoch}")
                inputs = batch.to(DEVICE)

                outputs = model(**inputs)
                loss = outputs.loss  # Reconstruction loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix_str(f"Loss: {total_loss / (i + 1):.4f}")

if __name__ == "__main__":
    main()