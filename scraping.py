from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys, ActionChains
import time
service = Service(
    executable_path=r"/Users/lestersanjuan/Desktop/ZottaGo/IloveTyping/chromedrivermac")
#service = Service(
 #   executable_path=r"C:\Users\Lester San Juan\Desktop\cs178\MonkeyTypeTyper\chromedriver.exe")
driver = webdriver.Chrome(service=service)

driver.implicitly_wait(2)

driver.get("https://snrsca.com/dashboard/catalog")

driver.implicitly_wait(2)


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


def find_all_items():
    product_names = driver.find_elements(By.CSS_SELECTOR, "h3>button")
    print(product_names)
    for word in product_names:
        print(word.text)


def add_to_cart(amount, item_name):
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
        product = driver.find_element(By.XPATH, f"//*[text()='{item_name}']")
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


while True:
    input_taker = input("press y whenever youre ready to find items\n")
    if input_taker == "y":
        find_all_items()
        add_to_cart(10, "Sunright 8684 uniform Apron 10 PCS")
    elif input_taker == "quit":
        break
    else:
        print('check')
