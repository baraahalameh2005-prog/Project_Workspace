"""Drive the try-on page like a user (Edge, headless): dress from two photos,
then a bag, then a skirt; screenshots of each step."""
import sys
import time

from playwright.sync_api import sync_playwright

T = "C:/Users/hp/NewAvatar/var/tmp/"
URL = "http://127.0.0.1:8010/viewer/?avatar=4c557cfb16074ee3aca2d5a2a406df02"
errors = []


def shot(page, name):
    page.screenshot(path=T + f"ui_{name}.png")
    print("shot", name, flush=True)


def wait_measured(page, n, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        busy = page.locator(".st.busy").count()
        done = page.locator(".st.ok").count() + page.locator(".st.err").count()
        if not busy and done >= n:
            return
        time.sleep(0.5)
    raise TimeoutError("photos not measured")


def wear(page, timeout=180):
    page.click("#wearBtn")
    end = time.time() + timeout
    while time.time() < end:
        text = page.inner_text("#wearStatus")
        if "جاهز" in text or "ما زبط" in text:
            print("  status:", text, flush=True)
            return text
        time.sleep(1)
    raise TimeoutError("wear timed out")


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True,
                                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(URL)
    page.evaluate("localStorage.clear()")
    page.reload()
    page.wait_for_timeout(4000)

    which = sys.argv[1:] or ["dress", "bag", "skirt"]
    for item in which:
        if item == "dress":
            page.click('.kind[data-kind="dress"]')
            page.click("#toPhotos")
            page.set_input_files("#filePick", [T + "a4_flat_front.jpg", T + "a4_flat_back.jpg"])
            wait_measured(page, 2)
            shot(page, "dress_photos")
            page.click("#toMeasures")
            page.wait_for_timeout(1500)
            page.hover('.field[data-name="shoulder_width"]')
            page.check("#opts input >> nth=0")
            page.check("#opts input >> nth=1")
            shot(page, "dress_measures")
            page.click("#toWear")
            wear(page)
            page.wait_for_timeout(4000)
            shot(page, "dress_worn")
        elif item == "nopaper":
            page.click('.kind[data-kind="dress"]')
            page.click("#toPhotos")
            page.set_input_files("#filePick", ["C:/Users/hp/NewAvatar/clothing/samples/flat_front.jpg",
                                               "C:/Users/hp/NewAvatar/clothing/samples/flat_back.jpg"])
            wait_measured(page, 2)
            shot(page, "np_photos")
            page.click("#toMeasures")
            page.wait_for_timeout(1000)
            shot(page, "np_scalebox")
            page.select_option("#scaleBox select", "length_from_shoulder")
            page.fill("#scaleBox input", "130")
            page.click("#scaleBox .btn.primary")
            page.wait_for_timeout(2500)
            shot(page, "np_measures")
            page.click("#toWear")
            wear(page)
            page.wait_for_timeout(4000)
            shot(page, "np_worn")
        elif item in ("bag", "belt", "hat", "glasses", "necklace"):
            page.click("#again") if page.locator("#again").is_visible() else None
            page.click('.kind[data-kind="accessory"]')
            page.click(f'.chip[data-acc="{item}"]')
            page.click("#toPhotos")
            page.evaluate("document.getElementById('filePick').value = ''")
            page.set_input_files("#filePick", [T + "a4_bag.jpg"])
            wait_measured(page, 1)
            page.click("#toMeasures")
            page.wait_for_timeout(1000)
            page.click("#toWear")
            wear(page)
            page.wait_for_timeout(2500)
            shot(page, f"{item}_worn")
        elif item == "skirt":
            page.click("#again") if page.locator("#again").is_visible() else None
            page.click('.kind[data-kind="skirt"]')
            page.click("#toPhotos")
            page.set_input_files("#filePick", [T + "a4_skirt.jpg"])
            wait_measured(page, 1)
            page.click("#toMeasures")
            page.wait_for_timeout(1000)
            page.click("#toWear")
            wear(page)
            page.wait_for_timeout(4000)
            shot(page, "skirt_worn")
    browser.close()
print("page errors:", errors or "none")
