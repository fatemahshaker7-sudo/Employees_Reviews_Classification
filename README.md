# Employee Review Classifier

[Employee Review Classifier Streamlit App](https://employeesreviewsclassifier.streamlit.app/)

Comparing pretrained Hugging Face models on Glassdoor employee reviews: sentiment (negative / neutral / positive) and zero-shot issue classification, with a Streamlit app.

## 1. Problem Definition
Manually reading thousands of employee reviews is not practical. This project tests whether pretrained models can (1) classify review sentiment and (2) detect the main issue (management, pay, work-life balance, career growth, culture) without fine-tuning.

## 2. Data Collection
- **Source:** [Glassdoor Employee Reviews (Kaggle)](https://www.kaggle.com/petersunga/google-amazon-facebook-employee-reviews)
- **Columns used:** `summary`, `pros`, `cons`, `overall-ratings`
- **Sample:** 100 random reviews (`random_state=42`), saved as `sample_100.csv`

## 3. Data Preparation
- Read with `encoding="latin-1"`; missing text replaced with empty strings.
- Text = `summary + pros + cons`.
- Sentiment labels from stars: 1-2 = `neg`, 3 = `neu`, 4-5 = `pos`.
- Issue labels: `classes` column filled in by hand (**30** reviews), predicted from `cons` text.

## 4. Model Selection
- **Sentiment (3 classes):** `cardiffnlp/twitter-roberta-base-sentiment-latest`, `lxyuan/distilbert-base-multilingual-cased-sentiments-student`, `finiteautomata/bertweet-base-sentiment-analysis`
- **Zero-shot (issues):** `typeform/distilbert-base-uncased-mnli`

## 5. Model Card Investigation
**Verify against each model's Hugging Face page.**
- RoBERTa and BERTweet were trained on tweets; DistilBERT is a smaller multilingual model.
- DistilBERT-MNLI was trained on MNLI for zero-shot classification.
- None were trained on employee reviews, so this is a domain-shift test.

## 6. Evaluation
Accuracy and macro F1, plus a confusion matrix per model.

| Task | Model | Accuracy | Macro F1 |
|---|---|---|---|
| Sentiment | RoBERTa | **0.71** | **0.55** |
| Sentiment | DistilBERT | **0.66** | **0.408** |
| Sentiment | BERTweet | **0.7** | **0.555** |
| Issues | DistilBERT-MNLI | **0.564** | **0.523** |

## 7. Error Analysis
Typical errors: neutral reviews predicted as positive or negative, mixed text ("Great pay, terrible management"), noisy star labels, and overlapping issues.

| Review (shortened) | True | Predicted | Why |
|---|---|---|---|
Okay, depends on your division and director Amazing benefits, especially insuran | neg | neu
Microsoft Rules Flexibility and good benefits and good food Nothing. Everything  | neu | pos
Life@Microsoft Benefits Salary Brand Microsoft works on various technologies and | neg | neu
Good Company to work with its a good company to grow with it has a great technol | neu | pos
Money isn't everything. Extremely intelligent group of employees. Employees are  | neg | neu
Good 1st time Job Benefits starting from the 1st day Stress, you have to meet th | neu | pos
SDE I think Amazon is the best company you can work if you cannot enter Google/F | pos | neg
Nice Headquarter in Ireland Cork B2B, I saw another side of customer service, an | pos | neu
Software Development Manager The people are fantastic. Fast moving and open envi | pos | neg
just so-so free working hours and life/work balance no belong sense and no headc | pos | neg
Brazil Operations Specialist Estable company Good culture Easy going Time off an | neu | pos

## 8. Model Comparison
RoBERTa model is the most accurate model, then BERTweet and lastly, DistilBERT

## 9. Limitations
- Small sample (100 reviews, fewer for issues).
- Sentiment labels come from star ratings, so some are noisy.
- Hand labels are subjective and made by one person.
- Models were not trained on employee reviews.
- Long reviews were truncated.
- Data covers a few tech companies from 2018.
- Zero-shot results depend on label wording.

## 10. Final Recommendation
RoBERTa is the best sentiment model, and the zero shot model needs fine tuning to be better.
