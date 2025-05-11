import re
import time
from datetime import datetime
from math import floor

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

from webdrive_config import setup_webdriver


# ajastus decoraattori ottaa function ja palauttaa funktion palautuksen, samalla ottaen aikaa function ajoajasta
def timing(f):
    def wrap(*args, **kw):
        time_start = time.time()
        result = f(*args, **kw)
        time_end = time.time()
        print(f"func: {f.__name__} took: {time_end - time_start:.3f} sec")
        return result

    return wrap


def get_html_from_url(url: str, driver: webdriver.Firefox):
    print("getting data from url")
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
    except Exception as e:
        print(f"An error occurred: {e}")
        driver.quit()
        exit()

    return soup


# hakee kokonais sivumäärän ja etsii oikean sivun
@timing
def get_correct_url(driver: webdriver.Firefox) -> str:
    url = "https://www.alko.fi/tuotteet/tuotelistaus?SearchTerm=*&PageSize=12&SortingAttribute=&PageNumber=1&SearchParameter=%26%40QueryTerm%3D*%26ContextCategoryUUID%3D6Q7AqG5uhTIAAAFVOmkI2BYA%26OnlineFlag%3D1"
    doc = get_html_from_url(url, driver)
    try:
        products = doc.find(class_="color-primary")
        products = str(*products)
    except Exception:
        print(doc)
        print("could not find the correct data on the website")
        exit()

    # regex hakee tuotemäärän
    product_count = re.findall("\d", products)
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
    price_data = doc.find_all(
        class_="js-price-container price-wrapper mc-price hide-for-list-view"
    )

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
    items = items.replace('"', "").strip()
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

            alcohol_per_l = round(float(alcohol) / adjusted_price, 3)

            df_data = {
                "Name": [name],
                "Alcohol": [alcohol],
                "Size": [size],
                "Price": [price],
                "Price_per_liter": [adjusted_price],
                "Alcohol_per_euro_per_liter": [alcohol_per_l],
            }

            new_data = pd.DataFrame(df_data)

            df = pd.concat([df, new_data], ignore_index=False)

            alcohol_dict.update(
                {name: [alcohol, size, price, f"{adjusted_price}", f"{alcohol_per_l}"]}
            )

        except Exception:
            # virheellinen tuote
            print(name, alcohol, size)

    sorted_alcohol_dict = {
        k: v for k, v in sorted(alcohol_dict.items(), key=lambda item: item[1][-1])
    }

    return sorted_alcohol_dict, df


def print_alcohol_data(sorted_alcohol_dict):
    for key in sorted_alcohol_dict:
        try:
            print(
                f"{key}: alkoholi % per litra per euro: {sorted_alcohol_dict[key][-1]}"
            )
        except Exception:
            pass

    print(f"lopullinen tuotemäärä: {len(sorted_alcohol_dict)}")
    # print("prices ", len(product_price))
    # print("alcohol dict ", len(alcohol_dict))
    # print(len(product_prices))


def main():
    try:
        driver = setup_webdriver(timeout_seconds=1800)
        url = get_correct_url(driver)
        print("getting data from Alko, please wait...")
        print(
            "seriously, it might take over 20 minutes, there is a lot of HTML that we need to download"
        )
        doc = get_html_from_url(url, driver)

    except Exception as e:
        print(f"couldnt get the necessary data from Alko: {e}")
    finally:
        driver.quit()

    print("website parsed")

    # print(doc.prettify())

    product_data, product_price = get_data_from_html(doc)

    sorted_alcohol_dict, dataframe = make_dict_from_data(product_data, product_price)

    # data tulostetaan konsoliin
    print_alcohol_data(sorted_alcohol_dict)

    # dataframesta tehdään kuvaaja
    dataframe.drop_duplicates(subset="Name", inplace=True)
    dataframe.sort_values("Alcohol_per_euro_per_liter", ascending=False, inplace=True)

    df_path = (
        f"Price_data/alcohol_prices_{datetime.now().month}_{datetime.now().year}.csv"
    )
    dataframe.to_csv(df_path)


if __name__ == "__main__":
    main()
