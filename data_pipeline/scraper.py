"""
Module 1: Scraper Component
Extracts book data from https://books.toscrape.com/, cleans and normalizes fields,
and converts GBP prices to INR.
"""

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Constants
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50
HTTP_TIMEOUT = 10

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

def fetch_html(url: str) -> str:
    """Fetch HTML content from a URL with timeout and error handling."""
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch URL {url}: {e}")
        return ""

def parse_rating(rating_classes: list) -> int:
    """Extract rating integer (1-5) from CSS class list."""
    for cls in rating_classes:
        if cls in RATING_MAP:
            return RATING_MAP[cls]
    return 0

def clean_and_transform_data(raw_books: list) -> pd.DataFrame:
    """
    Clean, impute missing values if needed, and transform raw scraped records into a Pandas DataFrame.
    Calculates price_inr using GBP_TO_INR_RATE constant.
    """
    if not raw_books:
        return pd.DataFrame(columns=["title", "price_gbp", "price_inr", "rating", "in_stock", "category"])

    df = pd.DataFrame(raw_books)

    # 1. Clean Title
    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace("â€™", "'", regex=False)
        .str.replace("â€œ", '"', regex=False)
        .str.replace("â€", '"', regex=False)
        .str.strip()
    )
    df = df[df["title"] != ""].copy()

    # 2. Clean Category
    df["category"] = df["category"].astype(str).str.strip()

    # 3. Clean Price GBP & Median Imputation if needed
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    if df["price_gbp"].isnull().any():
        valid_median = df["price_gbp"].median()
        print(f"[INFO] Imputing missing price_gbp values with median: {valid_median:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(valid_median)

    # Drop any row if price_gbp remains unrecoverable / NaN
    df = df.dropna(subset=["price_gbp"]).copy()
    df["price_gbp"] = df["price_gbp"].astype(float)

    # 4. Clean Rating (1 to 5)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
    # Ensure ratings are constrained to 1..5, fallback to median rating if 0
    invalid_ratings = (df["rating"] < 1) | (df["rating"] > 5)
    if invalid_ratings.any():
        median_rating = int(df.loc[~invalid_ratings, "rating"].median() or 3)
        print(f"[INFO] Adjusting invalid ratings to median rating: {median_rating}")
        df.loc[invalid_ratings, "rating"] = median_rating

    # 5. Clean in_stock (Boolean)
    df["in_stock"] = df["in_stock"].astype(bool)

    # 6. Price INR Transformation
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    return df

def scrape_books(min_books: int = 60, min_categories: int = 3) -> pd.DataFrame:
    """
    Main scraper function to retrieve books across categories until min_books and min_categories are reached.
    """
    index_html = fetch_html(BASE_URL + "index.html")
    if not index_html:
        raise RuntimeError("Unable to load books.toscrape.com home page.")

    soup = BeautifulSoup(index_html, "html.parser")
    cat_container = soup.find("div", class_="side_categories")
    if not cat_container:
        raise ValueError("Could not find side categories on books.toscrape.com.")

    cat_links = cat_container.find_all("a")[1:]  # skip root 'Books' link
    raw_books = []
    scraped_categories = set()

    for cat_tag in cat_links:
        cat_name = cat_tag.text.strip()
        cat_href = cat_tag["href"]
        cat_url = BASE_URL + cat_href

        current_page_url = cat_url
        while current_page_url:
            page_html = fetch_html(current_page_url)
            if not page_html:
                break

            page_soup = BeautifulSoup(page_html, "html.parser")
            pods = page_soup.find_all("article", class_="product_pod")

            for pod in pods:
                # Title
                a_tag = pod.h3.find("a")
                title = a_tag.get("title") if a_tag and a_tag.get("title") else (a_tag.text.strip() if a_tag else "")

                # Price GBP
                price_elem = pod.find("p", class_="price_color")
                price_gbp = None
                if price_elem:
                    match = re.search(r"[\d\.]+", price_elem.text)
                    if match:
                        price_gbp = float(match.group())

                # Rating
                rating_elem = pod.find("p", class_="star-rating")
                rating = parse_rating(rating_elem["class"]) if rating_elem else 0

                # Stock availability
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

            # Pagination check
            next_li = page_soup.find("li", class_="next")
            if next_li and next_li.find("a"):
                next_href = next_li.find("a")["href"]
                # Resolve relative link
                parent_url = current_page_url.rsplit("/", 1)[0]
                current_page_url = f"{parent_url}/{next_href}"
            else:
                current_page_url = None

        if len(raw_books) >= min_books and len(scraped_categories) >= min_categories:
            print(f"[INFO] Target reached: Scraped {len(raw_books)} books across {len(scraped_categories)} categories.")
            break

    df_clean = clean_and_transform_data(raw_books)

    # Validation guards
    if len(df_clean) < min_books:
        raise ValueError(f"Scraped dataset contains only {len(df_clean)} books, expected at least {min_books}.")

    if df_clean["category"].nunique() < min_categories:
        raise ValueError(f"Scraped dataset contains only {df_clean['category'].nunique()} categories, expected at least {min_categories}.")

    return df_clean

if __name__ == "__main__":
    df = scrape_books()
    print("Scraped DataFrame sample:")
    print(df.head())
    print("Summary:")
    print(df.info())
