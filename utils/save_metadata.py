import pickle
from dataset_loader import load_combined_datasets

def save_metadata():
    """Save metadata for FAISS index"""
    print("Loading datasets...")
    questions, answers, embedder, index, dataset_info = load_combined_datasets()
    
    metadata = {
        'questions': questions,
        'answers': answers,
        'embedder': embedder,
        'dataset_info': dataset_info
    }
    
    print("Saving metadata...")
    with open("../metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("Metadata saved successfully!")

if __name__ == "__main__":
    save_metadata() 