from bs4 import BeautifulSoup
import requests
import re
from math import *
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
import random

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

# ajastus decoraattori ottaa function ja palauttaa funktion palautuksen, samalla ottaen aikaa function ajoajasta
def timing(f):
    def wrap(*args, **kw):
        time_start = time.time()
        result = f(*args, **kw)
        time_end = time.time()
        print(f"func: {f.__name__} took: {time_end - time_start:.3f} sec")
        return result

    return wrap


def get_html_from_url(url, user_agent, timeout=None):
    headers = {'User-Agent': user_agent}
    result = requests.get(url, headers=headers, timeout=timeout)
    doc = BeautifulSoup(result.text, "html.parser")
    return doc


def get_html_from_url_selenium(url, user_agent):
    options = Options()
    options.add_argument(f"user-agent={user_agent}")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(random.uniform(2, 5))  # Random delay to mimic human

        # Simulate human-like scrolling
        for _ in range(random.randint(3, 5)):
            ActionChains(driver).move_by_offset(0, random.randint(100, 200)).perform()
            time.sleep(random.uniform(0.2, 0.5))

        # Get page source and parse it with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
    except Exception as e:
        print(f"An error occurred: {e}")
        soup = None
    finally:
        driver.quit()

    return soup

# hakee kokonais sivumäärän ja etsii oikean sivun
@timing
def get_correct_url(user_agent) -> str:

    url = "https://www.alko.fi/tuotteet/tuotelistaus?SearchTerm=*&PageSize=12&SortingAttribute=&PageNumber=1&SearchParameter=%26%40QueryTerm%3D*%26ContextCategoryUUID%3D6Q7AqG5uhTIAAAFVOmkI2BYA%26OnlineFlag%3D1"
    try:

        doc = get_html_from_url_selenium(url, user_agent)
        products = doc.find(class_="color-primary")
        products = str(*products)

    except Exception as e:
        print("couldn't load the page, please visit the website link before re-running the script:\n")
        print("https://www.alko.fi/tuotteet/tuotelistaus?SearchTerm=*&PageSize=12&SortingAttribute=&PageNumber=1&SearchParameter=%26%40QueryTerm%3D*%26ContextCategoryUUID%3D6Q7AqG5uhTIAAAFVOmkI2BYA%26OnlineFlag%3D1")

    # regex hakee tuotemäärän
    product_count = re.findall('\d', products)
    product_count = int("".join(product_count))
    print("tuotemäärä: ", product_count)

    # sivumäärä saadaan tuotemäärästä
    if product_count % 12 == 0:  # jos jako menee tasan, sivu ei lataa
        pages = floor(product_count / 12) - 1
    else:
        pages = floor(product_count / 12)

    print("sivumäärä: ", pages)

    url = f"https://www.alko.fi/tuotteet/tuotelistaus?SearchTerm=*&PageSize=12&SortingAttribute=&PageNumber={str(pages)}&SearchParameter=%26%40QueryTerm%3D*%26ContextCategoryUUID%3D6Q7AqG5uhTIAAAFVOmkI2BYA%26OnlineFlag%3D1"

    print("kaikkien tuotteiden url:")
    print(url)

    return url


@timing
def get_data_from_html(doc):

    prod_data = doc.find_all(class_="product-data-container")
    price_data = doc.find_all(class_="js-price-container price-wrapper mc-price hide-for-list-view")

    print("prices found")

    product_prices = []

    for price in price_data:
        price = str(price)

        new_price = price.split('"')[1].split(" €")[0]

        product_prices.extend(new_price)
    # print(new_price)

    product_data = (data["data-product-data"] for data in prod_data)
    product_price = [i["content"] for i in price_data]

    return product_data, product_price


# gets the wanted value in a string, for example '"alcohol": "25.0"' this returns 25.0
def split_items(list):
    items = list.split(":")[1]

    items = items.replace('"', '').strip()

    return items


def make_dict_from_data(product_data, product_price):
    alcohol_dict = {}
    df = pd.DataFrame()

    for i, product in enumerate(product_data):

        attributes = product.split(",")

        name = [s for s in attributes if "name" in s]
        alcohol = [s for s in attributes if "alcohol" in s]
        size = [s for s in attributes if "size" in s]

        try:
            name = split_items(*name)
            alcohol = split_items(*alcohol)
            size = split_items(*size)
            price = product_price[i]

            adjusted_price = round(float(price) / float(size), 3)

            alcohol_per_l = round(float(alcohol) / adjusted_price ,3)

            df_data = {'Name': [name], 'Alcohol': [alcohol], 'Size': [size], 'Price': [price],
                        'Price_per_liter': [adjusted_price], 'Alcohol_per_euro_per_liter': [alcohol_per_l]}

            new_data = pd.DataFrame(df_data)

            df = pd.concat([df, new_data], ignore_index=False)

            alcohol_dict.update({name: [alcohol, size, price, f"{adjusted_price}", f"{alcohol_per_l}"]})

        except Exception:
            # virheellinen tuote
            print(name, alcohol, size)

    sorted_alcohol_dict = {k: v for k, v in sorted(alcohol_dict.items(), key=lambda item: item[1][-1])}

    return sorted_alcohol_dict, df


def print_alcohol_data(sorted_alcohol_dict):
    for key in sorted_alcohol_dict:
        try:
            print(f"{key}: alkoholi % per litra per euro: {sorted_alcohol_dict[key][-1]}")
        except Exception as e:
            pass

    print(f"lopullinen tuotemäärä: {len(sorted_alcohol_dict)}")
    # print("prices ", len(product_price))
    # print("alcohol dict ", len(alcohol_dict))
    # print(len(product_prices))


def main():

    user_agent = "user-agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.115 Safari/537.36"

    url = get_correct_url(user_agent)

    print("getting data from Alko, please wait...")
    doc = get_html_from_url_selenium(url, user_agent)

    print("website parsed")

    # print(doc.prettify())
    
    product_data, product_price = get_data_from_html(doc)

    sorted_alcohol_dict, dataframe = make_dict_from_data(product_data, product_price)

    # data tulostetaan konsoliin
    print_alcohol_data(sorted_alcohol_dict)

    # dataframesta tehdään kuvaaja
    dataframe.drop_duplicates(subset="Name", inplace=True)
    dataframe.sort_values("Alcohol_per_euro_per_liter", ascending=False, inplace=True)

    path = f"Price_data/alcohol_prices_{datetime.now().month}_{datetime.now().year}.csv"

    dataframe.to_csv(path)

    # plotly avaa offline nettisivun top 30 parhaasta tuloksesta graaphina
    fig = px.bar(data_frame=dataframe.head(30), x="Name", y="Alcohol_per_euro_per_liter")
    fig.show()


if __name__ == '__main__':
    main()
