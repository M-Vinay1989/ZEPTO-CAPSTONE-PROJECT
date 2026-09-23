"""
Scraper module for books.toscrape.com
Extracts book information, cleans fields, and converts prices from GBP to INR.
"""

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fixed configuration constants
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50
HTTP_TIMEOUT = 10

# Mapping star rating words to integer numbers (1-5)
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

def fetch_html(url: str) -> str:
    """Fetch web page HTML content with timeout handling."""
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.RequestException as err:
        print(f"[Warning] Failed to fetch URL {url}: {err}")
        return ""

def parse_rating(rating_classes: list) -> int:
    """Convert CSS star-rating class list into integer rating (1-5)."""
    for cls in rating_classes:
        if cls in RATING_MAP:
            return RATING_MAP[cls]
    return 0

def clean_and_transform_data(raw_books: list) -> pd.DataFrame:
    """
    Cleans raw scraped book data and adds price_inr column.
    """
    if not raw_books:
        return pd.DataFrame(columns=["title", "price_gbp", "price_inr", "rating", "in_stock", "category"])

    df = pd.DataFrame(raw_books)

    # 1. Clean Title string and replace encoding artifacts
    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace("â€™", "'", regex=False)
        .str.replace("â€œ", '"', regex=False)
        .str.replace("â€", '"', regex=False)
        .str.strip()
    )
    df = df[df["title"] != ""].copy()

    # 2. Clean Category string
    df["category"] = df["category"].astype(str).str.strip()

    # 3. Clean Price GBP (convert to float, impute median if missing)
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        print(f"[Info] Imputing missing price_gbp values with median: {median_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    df = df.dropna(subset=["price_gbp"]).copy()
    df["price_gbp"] = df["price_gbp"].astype(float)

    # 4. Clean Rating (ensure rating is 1..5 int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
    invalid_mask = (df["rating"] < 1) | (df["rating"] > 5)
    if invalid_mask.any():
        median_rating = int(df.loc[~invalid_mask, "rating"].median() or 3)
        df.loc[invalid_mask, "rating"] = median_rating

    # 5. Clean in_stock availability boolean
    df["in_stock"] = df["in_stock"].astype(bool)

    # 6. Calculate price in INR (1 GBP = 105.50 INR)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    return df

def scrape_books(min_books: int = 60, min_categories: int = 3) -> pd.DataFrame:
    """
    Scrapes books across categories from books.toscrape.com until minimum requirements are met.
    """
    index_html = fetch_html(BASE_URL + "index.html")
    if not index_html:
        raise RuntimeError("Could not connect to books.toscrape.com")

    soup = BeautifulSoup(index_html, "html.parser")
    side_categories = soup.find("div", class_="side_categories")
    if not side_categories:
        raise ValueError("Side categories element not found on webpage")

    cat_links = side_categories.find_all("a")[1:]  # skip root 'Books' link
    raw_books = []
    scraped_categories = set()

    for cat_tag in cat_links:
        cat_name = cat_tag.text.strip()
        cat_url = BASE_URL + cat_tag["href"]

        current_page_url = cat_url
        while current_page_url:
            page_html = fetch_html(current_page_url)
            if not page_html:
                break

            page_soup = BeautifulSoup(page_html, "html.parser")
            pods = page_soup.find_all("article", class_="product_pod")

            for pod in pods:
                a_tag = pod.h3.find("a")
                title = a_tag.get("title") if a_tag and a_tag.get("title") else (a_tag.text.strip() if a_tag else "")

                price_elem = pod.find("p", class_="price_color")
                price_gbp = None
                if price_elem:
                    match = re.search(r"[\d\.]+", price_elem.text)
                    if match:
                        price_gbp = float(match.group())

                rating_elem = pod.find("p", class_="star-rating")
                rating = parse_rating(rating_elem["class"]) if rating_elem else 0

                avail_elem = pod.find("p", class_="instock availability")
                in_stock = "In stock" in avail_elem.text if avail_elem else False

                raw_books.append({
                    "title": title,
                    "price_gbp": price_gbp,
                    "rating": rating,
                    "in_stock": in_stock,
                    "category": cat_name
                })

            scraped_categories.add(cat_name)

            # Check for next page in category pagination
            next_li = page_soup.find("li", class_="next")
            if next_li and next_li.find("a"):
                next_href = next_li.find("a")["href"]
                parent_url = current_page_url.rsplit("/", 1)[0]
                current_page_url = f"{parent_url}/{next_href}"
            else:
                current_page_url = None

        if len(raw_books) >= min_books and len(scraped_categories) >= min_categories:
            print(f"[Info] Scraped {len(raw_books)} books across {len(scraped_categories)} categories.")
            break

    df_clean = clean_and_transform_data(raw_books)

    if len(df_clean) < min_books:
        raise ValueError(f"Insufficient books scraped ({len(df_clean)} < {min_books})")
    if df_clean["category"].nunique() < min_categories:
        raise ValueError(f"Insufficient categories scraped ({df_clean['category'].nunique()} < {min_categories})")

    return df_clean

if __name__ == "__main__":
    books_df = scrape_books()
    print("Sample scraped books:")
    print(books_df.head(5))
