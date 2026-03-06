from agents.scheme_rules import SCHEME_RULES

def check_eligibility(user_input):

    text = user_input.lower()

    eligible = []

    for scheme in SCHEME_RULES:

        matched = True

        for condition in scheme["conditions"]:

            if condition not in text:
                matched = False
                break

        if matched:
            eligible.append(scheme)

    return eligible