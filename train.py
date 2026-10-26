"""Train the model.  Usage:  python train.py [dataset.csv]
CSV needs a URL column and a label column (bad/good, phishing/legitimate, or 1/0)."""
import sys, os, random, string, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from features import extract

random.seed(7)

def load_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ucol = cols.get('url') or cols.get('urls')
    lcol = cols.get('label') or cols.get('class') or cols.get('status')
    y = df[lcol].astype(str).str.lower().isin(['bad', 'phishing', 'malicious', 'fake', '1']).astype(int)
    return df[ucol].astype(str).tolist(), y.tolist()

def synthetic(n=4000):
    """Fallback demo data so the project runs without a dataset. Use a real CSV for real accuracy."""
    good_domains = ['google.com','wikipedia.org','amazon.in','flipkart.com','irctc.co.in','sbi.co.in','hdfcbank.com',
                    'github.com','youtube.com','paytm.com','microsoft.com','apple.com','nytimes.com','bbc.com','linkedin.com']
    paths = ['', '/', '/about', '/products/item?id=42', '/news/2024/story', '/help/contact', '/search?q=phones']
    brands = ['paypal','amazon','sbi','hdfc','paytm','netflix','irctc','google','flipkart','icici']
    words = ['login','verify','secure','update','kyc','claim','prize','refund','account','confirm']
    tlds = ['xyz','top','click','icu','site','online','cfd','tk','live']
    rnd = lambda k: ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))
    X, y = [], []
    for _ in range(n):
        d = random.choice(good_domains)
        sub = random.choice(['', 'www.', 'mail.', 'support.'])
        X.append(random.choice(['https://', 'https://', 'http://']) + sub + d + random.choice(paths)); y.append(0)
    for _ in range(n):
        b, w, t = random.choice(brands), random.choice(words), random.choice(tlds)
        kind = random.randint(0, 4)
        if kind == 0: u = f'http://{b}-{w}-{rnd(4)}.{t}/{w}'
        elif kind == 1: u = f'https://{b}.{w}.{rnd(6)}.{t}/{w}/index.php?id={rnd(8)}'
        elif kind == 2: u = f'http://{random.randint(11,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}/{b}/{w}'
        elif kind == 3: u = f'http://bit.ly/{rnd(7)}'
        else: u = f'https://{b}{random.randint(10,99)}-{w}.{t}:{random.choice([8080,8443,3000])}/{rnd(12)}'
        X.append(u); y.append(1)
    return X, y

if __name__ == '__main__':
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        urls, labels = load_csv(sys.argv[1]); print(f'Loaded {len(urls)} URLs from {sys.argv[1]}')
    else:
        urls, labels = synthetic(); print('No dataset given: using synthetic demo data.')
    X = np.array([extract(u) for u in urls]); y = np.array(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=1)
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte), target_names=['safe', 'fake']))
    joblib.dump(model, 'model.joblib'); print('Saved model.joblib')
