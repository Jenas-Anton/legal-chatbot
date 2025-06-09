import time
import matplotlib.pyplot as plt
from utils.dataset_loader import load_combined_datasets
from utils.embedding_search import search_similar_cases

# Global list to keep track of timings
faiss_timings = []

def timed_search(query, index, questions, answers, k=5):
    start_time = time.time()
    results = search_similar_cases(query, index, questions, answers, k)
    end_time = time.time()
    duration = end_time - start_time

    faiss_timings.append((query[:50] + "...", duration))
    print(f"Query: '{query[:50]}...' took {duration:.3f} seconds.")
    return results

def main():
    print("Loading datasets...")
    questions, answers, embedder, index, dataset_info = load_combined_datasets()

    # Use 10 realistic legal queries
    queries = [
        "Intellectual property rights violation in software",
        "Contract breach between multinational corporations",
        "Land acquisition dispute involving government",
        "Insurance fraud in healthcare claims",
        "Cybercrime involving cryptocurrency theft",
        "Dispute over trademark infringement in branding",
        "Data privacy violation under GDPR",
        "Unlawful termination of employment without notice",
        "Housing society dispute over maintenance charges",
        "False advertising claims in online marketplaces"
    ]

    for query in queries:
        results = timed_search(query, index, questions, answers)
        if results:
            print(f"Top match question: {results[0]['question']}\n")

    # Plot bar chart of retrieval times per query
    if faiss_timings:
        labels, times = zip(*faiss_timings)

        plt.figure(figsize=(10, 6))
        plt.bar(labels, times, color='skyblue')
        plt.ylabel("Time (seconds)")
        plt.xlabel("Query")
        plt.title("FAISS Retrieval Time per Query")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Optional: Uncomment this section to show box plot if running each query multiple times
        """
        from collections import defaultdict
        timings_dict = defaultdict(list)
        for label, time_val in faiss_timings:
            timings_dict[label].append(time_val)

        plt.figure(figsize=(10, 6))
        plt.boxplot(timings_dict.values(), patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red'))
        plt.xticks(range(1, len(timings_dict)+1), list(timings_dict.keys()), rotation=45, ha='right')
        plt.ylabel("Retrieval Time (seconds)")
        plt.title("FAISS Retrieval Time per Query (Box Plot)")
        plt.tight_layout()
        plt.show()
        """
    else:
        print("No timings recorded.")

if __name__ == "__main__":
    main()
