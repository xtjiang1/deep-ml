import math
from collections import Counter

def calculate_entropy_target(labels: list, target_attr: str) -> float:
    if not labels:
        return 0.0
    counts = Counter(row[target_attr] for row in labels)
    counts_vals = list(counts.values())
    total = sum(counts_vals)
    probs = [x / total for x in counts_vals]
    entropy = - sum((px * math.log2(px)) for px in probs if px > 0)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    total_entropy = calculate_entropy_target(examples, target_attr)
    attr_subsets = {}
    for row in examples:
        val = row[attr]
        attr_subsets.setdefault(val, []).append(row)
    total_samples = len(examples)
    conditional_entropy = 0.0
    for val, subset in attr_subsets.items():
        weight = len(subset) / total_samples
        conditional_entropy += weight * calculate_entropy_target(subset, target_attr)
    return total_entropy - conditional_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    counts = Counter(row[target_attr] for row in examples)
    sorted_classes = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_classes[0][0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict | str:
    if not examples:
        return ""
    first_class = examples[0][target_attr]
    if all(row[target_attr] == first_class for row in examples):
        return first_class
    if not attributes:
        return majority_class(examples, target_attr)
    best_attr = None
    max_gain = -1.0
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > max_gain:
            max_gain = gain
            best_attr = attr
    tree = {best_attr: {}}
    all_values = sorted(list(set(row[best_attr] for row in examples)))
    for val in all_values:
        subset = [row for row in examples if row[best_attr] == val]
        remaining_attrs = [a for a in attributes if a != best_attr]
        subtree = learn_decision_tree(subset, remaining_attrs, target_attr)
        tree[best_attr][val] = subtree
    return tree
