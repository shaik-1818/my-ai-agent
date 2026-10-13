import re
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page


def _clean(value):
    return (value or "").strip()


def _selector_for_element(element, fallback_tag="input"):
    if element is None:
        return fallback_tag

    element_id = _clean(element.get_attribute("id"))
    if element_id:
        return f"#{element_id}"

    name = _clean(element.get_attribute("name"))
    if name:
        return f"[name='{name}']"

    placeholder = _clean(element.get_attribute("placeholder"))
    if placeholder:
        return f"[placeholder='{placeholder}']"

    href = _clean(element.get_attribute("href"))
    if href and fallback_tag == "a":
        return f"a[href='{href}']"

    return fallback_tag


def _build_element(tag, element, fallback_tag=None):
    tag_name = tag or fallback_tag or "input"
    selector = _selector_for_element(element, tag_name)

    text = _clean(element.inner_text())
    if not text:
        text = _clean(element.get_attribute("aria-label"))
    if not text:
        text = _clean(element.get_attribute("value"))

    details = {
        "tag": tag_name,
        "selector": selector,
        "id": _clean(element.get_attribute("id")),
        "name": _clean(element.get_attribute("name")),
        "type": _clean(element.get_attribute("type")),
        "placeholder": _clean(element.get_attribute("placeholder")),
        "value": _clean(element.get_attribute("value")),
        "text": text,
    }

    if tag_name == "textarea":
        details["value"] = _clean(element.input_value())
        if not text:
            details["text"] = details["value"]

    if tag_name == "select":
        details["value"] = _clean(element.input_value())

    if tag_name == "a":
        details["href"] = _clean(element.get_attribute("href"))

    return details


def browser_open(page: Page, url: str):

    print(f"\n🌐 Opening: {url}")

    page.goto(
        url,
        wait_until="domcontentloaded"
    )

    return {
        "success": True,
        "url": page.url,
        "title": page.title()
    }


def browser_read(page: Page):

    print("\n📖 Reading webpage...")

    # Prefer the main content area over the entire webpage.
     # Extract the most relevant visible page content.
        # Extract the most relevant visible webpage content.
    text = ""

    content_selectors = [
    "main article",
    "article",
    "[itemprop='articleBody']",
    "[role='main']",
    "main",
    ".problem-statement",
    ".question-content",
    ".content",
]

    for selector in content_selectors:
        try:
            locator = page.locator(selector)
            count = min(locator.count(), 5)

            for i in range(count):
                element = locator.nth(i)
                
                try:
                    element.evaluate("""
                        node => {
                            node.querySelectorAll(
                                'nav, footer, header, aside, [role="navigation"], ' +
                                '[role="banner"], [role="contentinfo"]'
                            ).forEach(el => el.remove());
                        }
                    """)
                except Exception:
                    pass

                if not element.is_visible():
                    continue

                candidate = _clean(
                        element.evaluate("""
                            node => {
                                const clone = node.cloneNode(true);

                                clone.querySelectorAll(
                                    'nav, footer, header, aside, ' +
                                    '[role="navigation"], [role="banner"], ' +
                                    '[role="contentinfo"], script, style, noscript'
                                ).forEach(el => el.remove());

                                return clone.innerText || '';
                            }
                        """)
                    )

                # Ignore short or empty content.
                if len(candidate) < 100:
                    continue

                # Prefer the longest meaningful content.
                if len(candidate) > len(text):
                    text = candidate

        except Exception:
            continue

    # Fall back to body text if no suitable main content exists.
    if len(text) < 100:
        try:
            text = _clean(
                page.locator("body").inner_text(timeout=5000)
            )
        except Exception:
            text = ""

    # Remove excessive whitespace and limit the content size.
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:8000]

    for selector in ("main", "article", "[role='main']"):
        try:
            locator = page.locator(selector).first

            if locator.count() > 0:
                candidate = _clean(locator.inner_text(timeout=2000))

                if len(candidate) > len(text):
                    text = candidate

        except Exception:
            continue

    # Fall back to the complete body if main content isn't available.
    if not text or len(text) < 100:
        try:
            text = _clean(page.locator("body").inner_text(timeout=3000))
        except Exception:
            text = ""

    links = []
    link_nodes = page.locator("a[href]")
    for i in range(min(link_nodes.count(), 20)):
        element = link_nodes.nth(i)
        links.append(_build_element("a", element, "a"))

    buttons = []
    button_nodes = page.locator(
        "button, [role='button'], input[type='button'], input[type='submit']"
    )
    for i in range(min(button_nodes.count(), 20)):
        element = button_nodes.nth(i)
        role = _clean(element.get_attribute("role"))
        element_type = _clean(element.get_attribute("type"))
        tag_name = element.evaluate("node => node.tagName.toLowerCase()")
        fallback = "button"
        if tag_name == "button":
            fallback = "button"
        elif role == "button":
            fallback = "[role='button']"
        elif element_type in {"button", "submit"}:
            fallback = f"input[type='{element_type}']"
        buttons.append(_build_element("button", element, fallback))

    inputs = []
    input_nodes = page.locator("input:not([type='hidden'])")
    for i in range(min(input_nodes.count(), 20)):
        element = input_nodes.nth(i)
        inputs.append(_build_element("input", element, "input"))

    textareas = []
    textarea_nodes = page.locator("textarea")
    for i in range(min(textarea_nodes.count(), 20)):
        element = textarea_nodes.nth(i)
        textareas.append(_build_element("textarea", element, "textarea"))

    editable_regions = []
    editable_nodes = page.locator(
        "[contenteditable='true'], [role='textbox']:not(input):not(textarea)"
    )
    for i in range(min(editable_nodes.count(), 20)):
        element = editable_nodes.nth(i)
        details = _build_element("editor", element, "[contenteditable='true']")
        details["role"] = _clean(element.get_attribute("role"))
        details["contenteditable"] = _clean(element.get_attribute("contenteditable"))
        editable_regions.append(details)

    editable_regions = textareas + editable_regions

    selects = []
    select_nodes = page.locator("select")
    for i in range(min(select_nodes.count(), 20)):
        element = select_nodes.nth(i)
        selects.append(_build_element("select", element, "select"))

    checkboxes = []
    checkbox_nodes = page.locator("input[type='checkbox']")
    for i in range(min(checkbox_nodes.count(), 20)):
        element = checkbox_nodes.nth(i)
        checkboxes.append(_build_element("checkbox", element, "input[type='checkbox']"))

    radio_buttons = []
    radio_nodes = page.locator("input[type='radio']")
    for i in range(min(radio_nodes.count(), 20)):
        element = radio_nodes.nth(i)
        radio_buttons.append(_build_element("radio", element, "input[type='radio']"))

    interactive_elements = []
    for category in (links, buttons, inputs, editable_regions, selects, checkboxes, radio_buttons):
        interactive_elements.extend(category)

    return {
        "success": True,
        "url": page.url,
        "title": page.title(),
        "page_url": page.url,
        "page_title": page.title(),
        "body_text": text[:4000],
        "links": links,
        "buttons": buttons,
        "inputs": inputs,
        "textareas": textareas,
        "editable_regions": editable_regions,
        "selects": selects,
        "checkboxes": checkboxes,
        "radio_buttons": radio_buttons,
        "interactive_elements": interactive_elements[:40],
    }


def browser_click(
    page: Page,
    selector: str
):

    try:

        print(
            f"\n🖱️ Clicking: {selector}"
        )

        page.locator(
            selector
        ).click(
            timeout=10000
        )

        return {
            "success": True,
            "url": page.url
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "url": page.url
        }


def browser_type(
    page: Page,
    selector: str,
    text: str
):

    try:

        print(
            f"\n⌨️ Typing into: {selector}"
        )

        page.locator(
            selector
        ).fill(
            text
        )

        return {
            "success": True
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


def browser_scroll(
    page: Page,
    direction: str = "down",
    amount: int = 500
):

    direction = (direction or "down").lower()

    if direction not in {"up", "down"}:
        return {
            "success": False,
            "error": "direction must be 'up' or 'down'",
            "url": page.url
        }

    try:

        amount = max(100, min(int(amount), 2000))

        delta = amount if direction == "down" else -amount

        page.evaluate(f"window.scrollBy(0, {delta})")

        page.wait_for_timeout(200)

        scroll_y = page.evaluate("window.scrollY")
        max_scroll = page.evaluate("document.documentElement.scrollHeight - window.innerHeight")

        return {
            "success": True,
            "url": page.url,
            "direction": direction,
            "amount": amount,
            "scroll_y": scroll_y,
            "max_scroll_y": max_scroll,
            "message": f"Scrolled {direction} by {amount}px."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "url": page.url
        }


def browser_press(
    page: Page,
    key: str,
    selector: str = None
):

    allowed_keys = {
        "Enter",
        "Escape",
        "Tab",
        "ArrowUp",
        "ArrowDown",
    }

    key_value = (key or "").strip()

    if not key_value:
        return {
            "success": False,
            "error": "No key provided.",
            "url": page.url
        }

    if key_value not in allowed_keys:
        return {
            "success": False,
            "error": f"Key '{key_value}' is not allowed. Use a safe browser key.",
            "url": page.url
        }

    try:

        if selector:
            page.locator(selector).focus()

        page.keyboard.press(key_value)

        return {
            "success": True,
            "url": page.url,
            "key": key_value,
            "selector": selector,
            "message": f"Pressed {key_value}."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "url": page.url
        }


def browser_screenshot(
    page: Page,
    name: str = "browser_screenshot",
    full_page: bool = False
):

    screenshots_dir = Path(__file__).resolve().parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in (name or "browser_screenshot")
    )
    if not safe_name:
        safe_name = "browser_screenshot"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    screenshot_path = screenshots_dir / f"{safe_name}_{timestamp}.png"

    try:

        page.screenshot(
            path=str(screenshot_path),
            full_page=full_page
        )

        return {
            "success": True,
            "url": page.url,
            "path": str(screenshot_path),
            "relative_path": f"screenshots/{screenshot_path.name}",
            "filename": screenshot_path.name,
            "message": f"Saved screenshot to {screenshot_path.name}."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "url": page.url
        }