def extract_user_profile(text):
    profile = {
        "farmer": False,
        "student": False,
        "low_income": False,
        "no_house": False
    }

    text = text.lower()

    if "farmer" in text:
        profile["farmer"] = True

    if "student" in text or "college" in text:
        profile["student"] = True

    if "poor" in text or "low income" in text:
        profile["low_income"] = True

    if "no house" in text:
        profile["no_house"] = True

    return profile