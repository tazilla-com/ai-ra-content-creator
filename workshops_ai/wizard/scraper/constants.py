# workshops_ai/wizard/scraper/constants.py
"""
Constants for web scraping - URL patterns for discovering services/about pages.
"""
from __future__ import annotations

# URL suffixes for services/products pages (multilingual)
SERVICES_SUFFIXES = [
    # Slovak
    "/sluzby",
    "/produkty",
    "/riesenia",
    "/ponuka",

    # Czech (with/without diacritics)
    "/služby",
    "/reseni",
    "/řešení",
    "/nabidka",
    "/nabídka",

    # English
    "/services",
    "/solutions",
    "/products",
    "/offer",
    "/portfolio",

    # German
    "/leistungen",
    "/produkte",
    "/losungen",
    "/lösungen",

    # Italian
    "/servizi",
    "/prodotti",
    "/soluzioni",

    # Spanish
    "/servicios",
    "/productos",
    "/soluciones",

    # Polish
    "/uslugi",
    "/usługi",
    "/rozwiazania",
    "/rozwiązania",
]

# URL suffixes for about/company pages (multilingual)
ABOUT_US_SUFFIXES = [
    # Slovak
    "/o-nas",
    "/o-spolocnosti",
    "/kto-sme",

    # English
    "/about",
    "/about-us",
    "/our-story",
    "/our-company",
    "/who-we-are",
    "/company",

    # Czech/Slovenian/Croatian
    "/o-spolecnosti",
    "/o-podjetju",
    "/o-nama",
    "/o-tvrtki",

    # German
    "/uber-uns",
    "/ueber-uns",
    "/wir-uber-uns",

    # Polish
    "/o-firmie",
    "/kim-jestesmy",

    # Hungarian
    "/rolunk",
    "/cegunkrol",

    # Italian
    "/chi-siamo",
    "/la-nostra-storia",

    # Spanish
    "/quienes-somos",
    "/sobre-nosotros",

    # French
    "/a-propos",
    "/qui-sommes-nous",
]

# HTML file extensions
HTML_EXTENSIONS = ["html", "htm", "php", "aspx", "asp", "jsp", "cfm"]
