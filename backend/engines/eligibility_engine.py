from scheme_rules import scheme_rules

def find_eligible_schemes(profile):

    profile = profile.lower()

    eligible = []

    for rule in SCHEME_RULES:

        match = True

        for cond in rule["conditions"]:

            if cond not in profile:
                match = False
                break

        if match:
            eligible.append(rule)

    return eligible