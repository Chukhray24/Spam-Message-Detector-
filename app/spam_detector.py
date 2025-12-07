import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path


DATA_PATH = Path("data/spam.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """
    Завантаження датасету.
    Для SMS Spam Collection з Kaggle потрібно взяти стовпці v1 (label) і v2 (text).
    """
    df = pd.read_csv(path, encoding="latin-1")   # для цього датасету потрібен latin-1
    # залишаємо тільки потрібні колонки
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # перетворюємо мітки в 0/1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


def train_test_vectorize(df: pd.DataFrame):
    """
    Розбиваємо дані на train/test + перетворюємо текст у числові ознаки TF-IDF.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # перетворювач тексту у TF-IDF вектори
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


def train_model(X_train_vec, y_train):
    """
    Навчаємо класифікатор (Multinomial Naive Bayes).
    """
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    return clf


def evaluate_model(clf, X_test_vec, y_test):
    """
    Оцінка моделі на тестовій вибірці.
    """
    y_pred = clf.predict(X_test_vec)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")


def save_artifacts(vectorizer, clf):
    """
    Зберігаємо модель і векторизатор на диск.
    """
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(clf, MODEL_DIR / "spam_model.joblib")
    print("Модель та векторизатор збережені в папці 'models/'")


def load_artifacts():
    """
    Завантажуємо модель та векторизатор з диска.
    """
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
    clf = joblib.load(MODEL_DIR / "spam_model.joblib")
    return vectorizer, clf


def predict_messages(messages, vectorizer, clf):
    """
    Прогноз для одного або кількох повідомлень.
    """
    if isinstance(messages, str):
        messages = [messages]

    X = vectorizer.transform(messages)
    preds = clf.predict(X)
    labels = ["ham" if p == 0 else "spam" for p in preds]
    return labels


def interactive_demo():
    """
    Простий консольний інтерфейс: введи текст — отримаєш SPAM/NOT SPAM.
    """
    vectorizer, clf = load_artifacts()
    print("Інтерактивний режим. Введіть повідомлення (або 'exit' для виходу).")
    while True:
        msg = input("> ")
        if msg.lower().strip() == "exit":
            break
        label = predict_messages(msg, vectorizer, clf)[0]
        print("Результат:", "SPAM" if label == "spam" else "NOT SPAM")


if __name__ == "__main__":
    # 1. завантажити дані
    df = load_data(DATA_PATH)

    # 2. підготувати train/test і TF-IDF
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = train_test_vectorize(df)

    # 3. навчити модель
    clf = train_model(X_train_vec, y_train)

    # 4. оцінити
    evaluate_model(clf, X_test_vec, y_test)

    # 5. зберегти артефакти
    save_artifacts(vectorizer, clf)

    # 6. запустити невелике демо
    interactive_demo()
