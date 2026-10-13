from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # Open Google
    page.goto("https://www.google.com")

    # Find Google's search box
    search_box = page.locator("textarea[name='q']")

    # Type the search
    search_box.fill("Python artificial intelligence")

    # Press Enter
    search_box.press("Enter")

    # Wait for results
    page.wait_for_load_state("domcontentloaded")

    print("Page title:", page.title())
    print("Current URL:", page.url)

    # Keep browser open for 5 seconds
    page.wait_for_timeout(5000)

    browser.close()