
# Bangla Sentiment Analysis under Adversarial Attacks

This repository implements **Bangla sentiment classification** using pretrained Transformer models (BERT, Electra, Multilingual BERT, etc.) and evaluates their robustness against **adversarial attacks**. The project falls under the domain of **Trustworthy AI**, specifically focusing on **Adversarial Robustness**.

## 🚀 Features

* Fine-tuning Bangla Transformer models (`sagorsarker/bangla-bert-base`, `csebuetnlp/banglabert`, `Kowsher/bangla-electra-bert-base`, etc.)
* Support for multiple Bangla sentiment datasets:

  * **BanglaBook** (large-scale Bangla book reviews)
  * **BigScience Bangla sentiment dataset**
  * Custom sample dataset (fallback)
* Evaluation with **accuracy, classification reports, and confusion matrix plots**
* Testing with **custom Bangla sentences**
* Generation of **adversarial examples** using [TextAttack](https://github.com/QData/TextAttack) (DeepWordBugGao2018 attack)
* Visualization of **training vs validation loss** and accuracy curves

---

## 📂 Project Structure

```
.
├── main.py                        # Main script for training & evaluation
├── requirements.txt               # Dependencies
├── results_bangla-bert-base/      # Trained model outputs
├── bangla_sentiment_bangla-bert-base/  # Saved fine-tuned model
├── confusion_matrix.png           # Confusion matrix plot
├── training_history.png           # Training vs Validation curves
├── bangla_adversarial_examples.csv # Generated adversarial attack logs
└── README.md                      # Project documentation
```

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/<your-username>/bangla-adversarial-sentiment.git
cd bangla-adversarial-sentiment

# Install dependencies
pip install -r requirements.txt
```

Or manually:

```bash
pip install transformers datasets torch textattack huggingface_hub scikit-learn matplotlib seaborn tensorflow==2.10.1
```

---

## 🧑‍💻 Usage

### 1. Run Training & Evaluation

```bash
python main.py
```

This will:

* Load datasets (BanglaBook, BigScience, or fallback sample dataset)
* Fine-tune the selected Transformer model
* Evaluate on the test set
* Save metrics, confusion matrix, and training history

### 2. Test with Custom Sentences

The model can be tested on Bangla texts after training. Example:

```python
test_texts = [
    "এই সিনেমাটি অসাধারণ ছিল! আমি খুবই পছন্দ করেছি।",   # Positive
    "খুবই খারাপ অভিজ্ঞতা। সম্পূর্ণ সময় নষ্ট হয়েছে।",   # Negative
    "বইটি ঠিক আছে। ভালো বা খারাপ কিছুই বিশেষ নয়।"       # Neutral
]
```

### 3. Generate Adversarial Examples

```python
generate_adversarial_examples(model, tokenizer, test_data, num_examples=5)
```

This creates adversarial samples and saves them into `bangla_adversarial_examples.csv`.

---

## 📊 Results

* **Confusion Matrix Example:**

![Confusion Matrix](confusion_matrix.png)

* **Training History Example:**

![Training History](training_history.png)

---

##  Trustworthy AI Perspective

Adversarial robustness is a key component of **Trustworthy AI**. This project explores:

* How Bangla NLP models perform under adversarial perturbations
* How robust they are against word-level attacks (DeepWordBug)
* The gap between **accuracy on clean data vs adversarial data**

---

## 📚 References

* HuggingFace Transformers: [https://huggingface.co/transformers](https://huggingface.co/transformers)
* TextAttack: [https://github.com/QData/TextAttack](https://github.com/QData/TextAttack)
* BanglaBERT: [https://huggingface.co/sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base)
* Bangla Electra: [https://huggingface.co/Kowsher/bangla-electra-bert-base](https://huggingface.co/Kowsher/bangla-electra-bert-base)

---

##  Author

**Md. Fariaz Islam**

* Electronics & Telecommunication Engineering, CUET
* Research Focus: NLP, Trustworthy AI, Adversarial Robustness

---

 *This project contributes towards building safer and more reliable AI systems for Bangla language understanding under adversarial settings.*

---


