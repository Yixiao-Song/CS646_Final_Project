class RetrievalEval():
    def __init__(self, retrieved_results, ground_truth, k=5):
        """Initialize the evaluator with retrieved results and ground truth.

        Args:
            - retrieved_results (list of list): A list where each sublist
            contains the top-k retrieved document IDs for a query.
            - ground_truth (list of list): A list where each sublist contains
            the relevant document IDs for a query.
        """
        self.retrieved_results = retrieved_results
        self.ground_truth = ground_truth
        self.num_queries = len(self.retrieved_results)
        self.k = k

    def recall_at_k(self):
        """Calculate the recall at k for the retrieved results.
        
        Returns:
            - recall (float): The recall at k for the retrieved results.
        """
        total_recall = 0

        for retrieved, ground in zip(self.retrieved_results, self.ground_truth):
            # Compute the intersection of the retrieved and ground truth
            intersection = set(retrieved[:self.k]) & set(ground)
            recall = len(intersection) / len(ground) if len(ground) > 0 else 0
            total_recall += recall

        return total_recall / self.num_queries
    
    def precision_at_k(self):
        """Calculate the precision at k for the retrieved results.
        
        Returns:
            - precision (float): The precision at k for the retrieved results.
        """
        total_precision = 0

        for retrieved, ground in zip(self.retrieved_results, self.ground_truth):
            # Compute the intersection of the retrieved and ground truth
            intersection = set(retrieved[:self.k]) & set(ground)
            precision = len(intersection) / self.k
            total_precision += precision

        return total_precision / self.num_queries
    
    def f1_at_k(self):
        """Calculate the F1 score at k for the retrieved results.
        
        Returns:
            - f1_score (float): The F1 score at k for the retrieved results.
        """
        total_f1_score = 0

        for retrieved, ground in zip(self.retrieved_results, self.ground_truth):
            # Compute the intersection of the retrieved and ground truth
            intersection = set(retrieved[:self.k]) & set(ground)
            precision = len(intersection) / self.k
            recall = len(intersection) / len(ground) if len(ground) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) \
                if precision + recall > 0 else 0
            total_f1_score += f1_score

        return total_f1_score / self.num_queries
    
    def mean_average_precision(self):
        """Calculate the mean average precision for the retrieved results.
        
        Returns:
            - mean_average_precision (float): The mean average precision for
            the retrieved results.
        """
        total_ap = 0

        for retrieved, ground in zip(self.retrieved_results, self.ground_truth):
            ground = set(ground)
            num_ground = len(ground)

            if num_ground == 0:
                continue

            ap = 0
            num_hits = 0

            for rank, url in enumerate(retrieved[:self.k]):
                if url in ground:
                    num_hits += 1
                    ap += num_hits / (rank + 1)
        
            ap /= num_ground
            total_ap += ap

        return total_ap / self.num_queries
