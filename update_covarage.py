import re

def extract():
    try:
        with open("coverage.txt", "rb") as f:
            raw = f.read()

        ## May need to change/adjust for different platforms
        for enc in ("utf-16", "latin1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue

        else:
            print("Can not read coverage.txt")
            return None
        
    except FileNotFoundError:
        print("coverage.txt does not exist")
        return None


def update_readme(report):
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()


    ### Used AI for the regex matching
    updated = re.sub(
        r"\s*-- COVERAGE_REPORT --.*?```",
        f"\n-- COVERAGE_REPORT --\n{report.strip()}\n```",
        content,
        flags=re.DOTALL
    )
    ### End use 

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    report = extract()
    if report:
        update_readme(report)
    else:
        print("Failed to update coverage report in README.md")