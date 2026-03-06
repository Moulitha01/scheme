from scheme_rules import scheme_rules

def find_eligible_schemes(profile):
    results = []

    for scheme, data in scheme_rules.items():
        conditions = data["eligible_if"]

        for c in conditions:
            if profile.get(c):
                results.append((scheme, data["benefit"]))
                break

    return results