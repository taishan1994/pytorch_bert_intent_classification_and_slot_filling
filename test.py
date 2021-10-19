from seqeval.metrics import accuracy_score as ner_accuracy_score
from seqeval.metrics import precision_score as ner_precision_score
from seqeval.metrics import recall_score as ner_recall_score
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.metrics import classification_report as ner_classification_report

y_pred = [['B-name', 'I-name', 'I-name', 'I-name'], ['B-ingredient', 'I-ingredient', 'O', 'O', 'O', 'O'], ['O', 'B-startLoc_city', 'I-Dest', 'O', 'O', 'O', 'O', 'B-endLoc_city', 'I-endLoc_city', 'O', 'O', 'O'], ['B-keyword', 'I-keyword', 'O'], ['O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'B-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content'], ['B-name', 'I-name', 'I-name', 'I-name', 'I-name'], ['O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'B-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'B-category', 'I-name', 'O', 'O', 'O', 'B-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content'], ['O', 'O', 'O', 'B-name', 'I-name']]
y_true = [['B-name', 'I-name', 'I-name', 'I-name'], ['B-ingredient', 'I-ingredient', 'O', 'O', 'O', 'O'], ['O', 'B-Src', 'I-Src', 'O', 'O', 'O', 'O', 'B-Dest', 'I-Dest', 'O', 'O', 'O'], ['B-keyword', 'I-keyword', 'O'], ['O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'B-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content'], ['B-name', 'I-name', 'I-name', 'I-name', 'I-name'], ['O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'B-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'B-teleOperator', 'I-teleOperator', 'O', 'O', 'O', 'B-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content'], ['O', 'O', 'O', 'B-name', 'I-name']]


acc = ner_accuracy_score(y_true, y_pred)
precision = ner_precision_score(y_true, y_pred)
recall = ner_recall_score(y_true, y_pred)
f1 = ner_f1_score(y_true, y_pred)
report = ner_classification_report(y_true, y_pred)

print(acc, precision, recall, f1)
print(report)