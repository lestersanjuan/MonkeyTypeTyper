import datetime
import traceback
from google.oauth2.service_account import Credentials
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys, ActionChains
import gspread
import time
from item_warehouse_dictionary import all_items_warehouse
# service = Service(
#   executable_path=r"/Users/lestersanjuan/Desktop/ZottaGo/IloveTyping/chromedrivermac")
service = Service(
    executable_path=r"C:\Users\Lester San Juan\Desktop\cs178\MonkeyTypeTyper\chromedriver.exe")
driver = webdriver.Chrome(service=service)
driver.implicitly_wait(2)
driver.get("https://snrsca.com/dashboard/catalog")
# driver.get("https://www.costcobusinessdelivery.com/")
driver.implicitly_wait(2)


test = "chester"

scopes = [
    "https://www.googleapis.com/auth/spreadsheets"
]

creds = Credentials.from_service_account_file(
    "credentials.json", scopes=scopes)
client = gspread.authorize(creds)

testing_id = "14mC696rRzj50gtN5VHTh1n-b8AjCPlKw8er4eG7IIvM"


def time_to_cheat():
    typed_words = driver.find_elements(By.ID, "words")
    about_to_type = driver.find_elements(By.CLASS_NAME, "word")
    print("These are the type words over here=> ", typed_words,
          "\n about to be typed here =>", about_to_type)

    for w in about_to_type:
        try:
            for s in w.text:
                time.sleep(0.001)
                ActionChains(driver).send_keys(s).perform()
        except:
            print("it breaks lol")
            continue
        ActionChains(driver).send_keys(" ").perform()

        print(w.text)

        to_type = driver.find_elements_by_css_selector(By.CLASS_NAME, "word")
        about_to_type.append(to_type)
    print(about_to_type)


def normalize_string(string):
    return string.replace("/", "").replace("(", "").replace(")", "").replace("=", "")


def get_all_order_by_box_and_names(sheet_id="14mC696rRzj50gtN5VHTh1n-b8AjCPlKw8er4eG7IIvM"):
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.get_worksheet(1)
    name_of_items = worksheet.col_values(1)
    number_to_order = worksheet.col_values(6)

    for name, number in list(zip(name_of_items, number_to_order))[2:]:
        name = name.strip()
        number = number.strip()
        print(f"The name of the item is, {name} while we order {number} boxes")
        try:
            if number != "":
                amount = int(number)
                add_to_cart_warehouse(amount, all_items_warehouse[name])
        except Exception as e:
            print(f"This did not work {name}, {number}")
            print(f"Error: {e}")
            traceback.print_exc()

        if name == "Chocolate Powder (20 bags)":
            break


def order_costco(sheet_id="14mC696rRzj50gtN5VHTh1n-b8AjCPlKw8er4eG7IIvM"):
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.get_worksheet(1)
    name_of_items = worksheet.col_values(1)
    number_to_order = worksheet.col_values(6)
    printing = False
    for name, number in list(zip(name_of_items, number_to_order))[2:]:
        name = name.strip()
        number = number.strip()
        if name == "Whole Milk (2/box)":
            printing = True
        if printing:
            try:
                input = driver.find_element(By.ID, "search-field")
                input.send_keys(normalize_string(name))
                driver.find_element(
                    By.CSS_SELECTOR, ".flex-child>button").click()
                driver.implicitly_wait(5)
                # Re-find the input field after DOM update
                input = driver.find_element(By.ID, "search-field")
                input.send_keys(Keys.CONTROL + "a")
                time.sleep(1)
                input.send_keys(Keys.DELETE)
            except Exception as e:
                print(f"Error: {e}")

        if name == "<>":
            break


def find_all_items():
    product_names = driver.find_elements(By.CSS_SELECTOR, "h3>button")
    print(product_names)
    for word in product_names:
        print(word.text)


def add_to_cart_warehouse(amount, item_name):
    if type(amount) != str:
        amount = str(amount)

    try:
        """
        <div>
            <div>
                <div>
                    <div>
                    item name
                    </div>
                </div>
            </div>
            <button>Add to Order</button>
        </div>

        """
        product = driver.find_element(
            By.XPATH, f"//*[contains(normalize-space(text()), '{item_name.strip()}')]")
        parent = product.find_element(By.XPATH, "../../..")
        cell = product.find_element(By.XPATH, "./../../..")
        # Find the input child directly under the parent
        input_child = parent.find_element(By.XPATH, ".//input")
        input_child.clear()
        input_child.send_keys(amount)
        # Find all buttons in the cell
        buttons = cell.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            btn_text = btn.text.strip()
            if btn_text == "Add to Order":
                btn.click()
                print("Clicked 'Add to Order'")
                break
            elif btn_text == "Out of Stock":
                print("Item is out of stock.")
                break
        else:
            print("No relevant button found.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    while True:
        input_taker = input("press y whenever youre ready to find items\n")
        if input_taker == "y":
            # find_all_items()
            get_all_order_by_box_and_names()
            #order_costco()
        elif input_taker == "quit":
            break
        else:
            print('check')
