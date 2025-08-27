import torch
from preprocessing import tokenizer, test_df
from training import trained_model, device

def predict_sample(model, tokenizer, text, device):
    model.eval()
    encodings = tokenizer.encode_plus(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return predicted_class, probs.squeeze().cpu().numpy()

# Get some sample texts from the test set
sample_indices = [0, 5, 10, 15, 20]
sample_texts = test_df.iloc[sample_indices]['combined_text'].tolist()
sample_labels = test_df.iloc[sample_indices]['label'].tolist()

print("Sample Predictions:")
for i, text in enumerate(sample_texts):
    predicted_label, probabilities = predict_sample(trained_model, tokenizer, text, device)
    print(f"Text: {text[:100]}...")
    print(f"Actual Label: {sample_labels[i]}, Predicted Label: {predicted_label}, Probabilities: {probabilities}")
    print("-" * 20)