from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:

        browser = p.chromium.launch(headless=False)

        page = browser.new_page()

        # Use a simple website for testing
        page.goto(
            "https://example.com",
            wait_until="domcontentloaded"
        )

        print("\n==============================")
        print("PAGE TITLE")
        print("==============================")

        print(page.title())

        print("\n==============================")
        print("PAGE URL")
        print("==============================")

        print(page.url)

        print("\n==============================")
        print("PAGE CONTENT")
        print("==============================")

        text = page.locator("body").inner_text()

        print(text)

        page.wait_for_timeout(5000)

        browser.close()


if __name__ == "__main__":
    main()