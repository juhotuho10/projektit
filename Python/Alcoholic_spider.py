from bs4 import BeautifulSoup
import requests
import re
from math import *

# hakee kokonais sivumäärän

url = "https://www.alko.fi/tuotteet/tuotelistaus?SearchTerm=*&PageSize=12&SortingAttribute=&PageNumber=1&SearchParameter=%26%40QueryTerm%3D*%26ContextCategoryUUID%3D6Q7AqG5uhTIAAAFVOmkI2BYA%26OnlineFlag%3D1"
result = requests.get(url)
doc = BeautifulSoup(result.text, "html.parser")

products = doc.find(class_="color-primary")

products = str(*products)

print(products)

# regex hakee tuotemäärän
product_count = re.findall('\d', products)
product_count = int("".join(product_count))

print("tuotemäärä: ", product_count)

# sivumäärä saadaan tuotemäärästä
if product_count % 12 == 0:
    pages = floor(product_count / 12)-1
else:
    pages = floor(product_count / 12)

print("sivumäärä: ", pages)

# lataa Alkon nettisivut kaikki sivumäärät mukaanlukien

url = "https://www.alko.fi/tuotteet/tuotelistaus?SearchTerm=*&PageSize=12&SortingAttribute=&PageNumber=" + str(pages-1) + "&SearchParameter=%26%40QueryTerm%3D*%26ContextCategoryUUID%3D6Q7AqG5uhTIAAAFVOmkI2BYA%26OnlineFlag%3D1"

print(url)

result = requests.get(url)

doc = BeautifulSoup(result.text, "html.parser")

print("website parsed")

# print(doc.prettify())

prices = doc.find_all(class_="product-data-container")
price_data = doc.find_all(class_="js-price-container price-wrapper mc-price hide-for-list-view")

print("prices found")

product_prices = []

for price in price_data:
    price = str(price)

    new_price = price.split('"')[1].split(" €")[0]

    product_prices.extend(new_price)
# print(new_price)


product_data = (price["data-product-data"] for price in prices)
product_price = [i["content"] for i in price_data]

alcohol_dict = {}


def split_items(list):
    names = list.split(":")[1]

    names = names.replace('"', '').strip()

    return names


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

        try:
            adjusted_price = float(price) / float(size)

            alcohol_per_l = float(alcohol) / adjusted_price

            alcohol_dict.update({name: [alcohol, size, price, f"{adjusted_price:.3f}", f"{alcohol_per_l:.3f}"]})
        except:
            # virheellinen tuote
            print(name, price, size)
    except:
        # virheellinen tuote
        print(name, alcohol, size)

sorted_alcohol_dict = {k: v for k, v in sorted(alcohol_dict.items(), key=lambda item: item[1][-1])}

for key in sorted_alcohol_dict:
    # print(i, sorted_alcohol_dict[i])

    print(f"{key}: alkoholi % per litra per euro: {sorted_alcohol_dict[key][-1]}")

print("tuotemäärä: ", len(sorted_alcohol_dict))
# print("prices ", len(product_price))
# print("alcohol dict ", len(alcohol_dict))
# print(len(product_prices))
