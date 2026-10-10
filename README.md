# Course Reviews Sentiment Analysis with Naive Bayes

A natural language processing (NLP) project that reads online course reviews and predicts whether each one is **good** or **bad**, using TF-IDF and a Multinomial Naive Bayes classifier.

## The goal

Course reviews are rated from 1 to 5 stars, but the comment text is what explains *why*. This project tests whether a simple, fast model can read a comment and predict the sentiment behind it, so negative feedback can be spotted automatically.

## Dataset

About 10,800 student reviews of online courses, each with:

| Column | Description |
|--------|-------------|
| `course_name` | Course being reviewed |
| `lecture_name` | Lecture being reviewed |
| `review_rating` | Rating from 1 to 5 |
| `review_comment` | Free-text comment |

A separate test set of 150 reviews was kept aside for a final check.

## What I did

1. **Loaded and cleaned the data.** I skipped malformed CSV rows, dropped reviews with a missing rating, converted text to lowercase and removed punctuation.
2. **Explored the data.**
   - Ratings are heavily skewed: the average is 4.84 and the median is 5.
   - Longer comments tend to come with lower ratings (correlation of about -0.28).
   - Positive words ("great", "good", "love") appear in over half of the reviews. Negative words are rare.
3. **Built the model.** A scikit-learn pipeline turns text into TF-IDF features (ignoring English stop words) and feeds them to a Multinomial Naive Bayes classifier.
4. **Tried predicting the exact star rating (1 to 5).** After balancing the classes, accuracy was about 74%. The model struggled with the rare 1 and 2 star reviews (recall of only 19% and 28%).
5. **Simplified to two classes:** *good* (4 to 5 stars) and *bad* (1 to 3 stars).
6. **Found that accuracy was misleading.** About 96% of reviews are good, so the first binary model scored 96.2% accuracy by predicting "good" for everything. It caught none of the bad reviews.
7. **Balanced the classes** by oversampling the bad reviews, which gave much more even results.
8. **Tested on the separate set of unseen reviews.**

## Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| 5-class (stars) | ~74% | Weak on 1 and 2 star reviews |
| Binary, unbalanced | 96.2% | Misleading: never predicts "bad" |
| Binary, balanced | 90.5% | Precision and recall around 0.9 for both classes |
| **Binary, unseen test set (150 reviews)** | **84.7%** | Catches 77% of bad reviews, with 33% precision |

## Conclusion

- A simple TF-IDF + Naive Bayes model is a fast baseline for sentiment analysis. On unseen reviews it correctly flagged about 3 out of 4 bad reviews.
- **Accuracy alone can be misleading** on imbalanced data. Precision, recall and the confusion matrix show what is really happening.
- The model raises many false alarms (33% precision on "bad"), because bad reviews are rare and the vocabulary of good and bad reviews overlaps.

## Limitations and next steps

- Oversampling was done **before** splitting into training and testing data, so duplicated bad reviews appear in both sets. The 90.5% score is therefore optimistic, and the **84.7% on the unseen test set is the more realistic figure**.
- The test set has only 13 bad reviews, so its metrics are noisy.
- Next steps: split first and oversample only the training data, try cross-validation, class weights, bigrams, and models such as logistic regression.

## How to run

1. Install the requirements:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Keep `user_courses_review_09_2023.csv` and `user_courses_review_test_set.csv` in the same folder as the notebook
3. Open `Course_Reviews_Sentiment_Analysis.ipynb` in Jupyter and run the cells from top to bottom

## Tech used

Python, pandas, scikit-learn (TF-IDF, Multinomial Naive Bayes), Matplotlib, Seaborn, Jupyter Notebook
