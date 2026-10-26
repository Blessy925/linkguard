"""Turns a URL into numbers the ML model can learn from (shared by train.py and app.py)."""
import math, re
from urllib.parse import urlparse

SHORTENERS = {'bit.ly','tinyurl.com','t.co','goo.gl','is.gd','cutt.ly','rb.gy','ow.ly','shorturl.at','tiny.cc','s.id'}
BAD_TLDS = {'xyz','top','click','link','work','support','buzz','icu','cyou','monster','rest','fit','live','shop','online','site','cfd','sbs','gq','tk','ml','ga','cf'}
KEYWORDS = ['login','signin','verify','secure','update','account','suspend','blocked','kyc','otp','password',
            'claim','prize','reward','gift','free','bonus','lottery','winner','refund','urgent','confirm','wallet','recharge','upi']
BRANDS = {'paypal':'paypal.com','amazon':'amazon.','flipkart':'flipkart.com','google':'google.','facebook':'facebook.com',
          'instagram':'instagram.com','whatsapp':'whatsapp.com','microsoft':'microsoft.com','apple':'apple.com',
          'netflix':'netflix.com','paytm':'paytm.com','phonepe':'phonepe.com','sbi':'sbi.co.in','hdfc':'hdfcbank.com',
          'icici':'icicibank.com','axis':'axisbank.com','irctc':'irctc.co.in'}

FEATURE_NAMES = ['url_len','host_len','path_len','dots','hyphens','digits_ratio','at_sign','is_ip','https','subdomains',
                 'params','host_entropy','bad_tld','shortener','keywords','brand_spoof','punycode','odd_port',
                 'double_slash','mixed_digits','upper_ratio','special_chars']

def normalize(url):
    url = url.strip()
    if not re.match(r'^[a-z][a-z0-9+.-]*://', url, re.I):
        url = 'http://' + url
    return url

def _entropy(s):
    if not s: return 0.0
    return -sum((s.count(c)/len(s))*math.log2(s.count(c)/len(s)) for c in set(s))

def extract(url):
    url = normalize(url)
    u = urlparse(url)
    host = (u.hostname or '').lower()
    h = host[4:] if host.startswith('www.') else host
    parts = h.split('.')
    tld = parts[-1] if parts else ''
    root = '.'.join(parts[-2:])
    label = parts[-2] if len(parts) >= 2 else h
    full = (u.path + '?' + u.query).lower()
    spoof = 0
    for b, official in BRANDS.items():
        if b in h and official not in h:
            spoof = 1
    return [
        len(url), len(h), len(u.path), h.count('.'), h.count('-'),
        sum(c.isdigit() for c in url) / max(len(url), 1),
        int('@' in url), int(bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', h))),
        int(u.scheme == 'https'), max(len(parts) - 2, 0), len(u.query.split('&')) if u.query else 0,
        _entropy(h), int(tld in BAD_TLDS), int(root in SHORTENERS),
        sum(k in (h + full) for k in KEYWORDS), spoof, int('xn--' in h),
        int(bool(u.port) and u.port not in (80, 443)), int('//' in u.path),
        int(any(c.isdigit() for c in label) and any(c.isalpha() for c in label)),
        sum(c.isupper() for c in url) / max(len(url), 1),
        sum(not c.isalnum() and c not in './:-_' for c in url),
    ]

def reasons(url):
    """Plain-English findings shown to the user."""
    f = dict(zip(FEATURE_NAMES, extract(url)))
    r = []
    if not f['https']: r.append('Not secure: no HTTPS')
    if f['is_ip']: r.append('Uses an IP address, not a name')
    if f['at_sign']: r.append('Hidden redirect using “@”')
    if f['punycode']: r.append('Look-alike characters in domain')
    if f['shortener']: r.append('Shortened link hides real site')
    if f['bad_tld']: r.append('Risky domain ending')
    if f['brand_spoof']: r.append('Pretends to be a known brand')
    if f['keywords']: r.append('Bait words in link')
    if f['subdomains'] > 2: r.append('Too many subdomains')
    if f['hyphens'] >= 2: r.append('Many hyphens in domain')
    if f['odd_port']: r.append('Odd port number')
    if f['url_len'] > 100: r.append('Unusually long link')
    return r
