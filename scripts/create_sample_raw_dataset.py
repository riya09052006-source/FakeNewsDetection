import os
import random
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FAKE_PATH = RAW_DIR / "Fake.csv"
TRUE_PATH = RAW_DIR / "True.csv"

if FAKE_PATH.exists() and TRUE_PATH.exists():
    print("Fake.csv and True.csv already exist.")
else:
    print("Generating comprehensive dataset for Fake.csv and True.csv...")
    
    real_subjects = ["politicsNews", "worldnews"]
    fake_subjects = ["News", "politics", "Government News", "left-news", "US_News", "Middle-East"]

    fake_titles = [
        "BREAKING: Shocking Secret Uncovered About High-Level Officials!",
        "MUST SEE: Whistleblower Reveals Unbelievable Scandal In City Council!",
        "UNBELIEVABLE: Major Media Outlets Suppress Huge News Story!",
        "BOMBSHELL: Proof Of Mass Cover-Up Discovered By Independent Researchers!",
        "YOU WON'T BELIEVE WHAT HAPPENED: Leaked Documents Expose Hidden Agenda!",
        "SHOCK REPORT: Insiders Claim Mysterious Event Changed Everything Overnight!",
        "EXPOSED: Top Bureaucrats Caught In Massive Corruption Plot!",
        "URGENT: Secret Plan To Alter Public Health Guidelines Discovered!",
        "WATCH: Explosive Video Reveals Untold Truth Behind National Controversy!",
        "ALERT: Millions Affected As Unverified Claims Circulate Online!"
    ]
    
    real_titles = [
        "WASHINGTON (Reuters) - Senate Passes Bipartisan Infrastructure Investment Bill",
        "LONDON (Reuters) - Central Bank Raises Interest Rates To Combat Inflation",
        "TOKYO (Reuters) - Tech Giants Announce Partnership On Next-Gen Semiconductor Tech",
        "PARIS (Reuters) - European Union Leaders Meet To Discuss Climate Policy Goals",
        "BEIJING (Reuters) - Trade Representatives Hold Bilateral Negotiations In Capital",
        "GENEVA (Reuters) - World Health Organization Updates Global Influenza Guidance",
        "NEW YORK (Reuters) - Wall Street Stocks Steady Ahead Of Federal Reserve Decision",
        "BERLIN (Reuters) - Chancellor Pledges Increased Funding For Renewable Energy",
        "CANBERRA (Reuters) - Australia Signs Bilateral Security Accord With Pacific Island Nation",
        "OTTAWA (Reuters) - Canadian Government Outlines New Environmental Regulations"
    ]

    fake_texts = [
        "An anonymous whistleblower has stepped forward with startling claims regarding a hidden conspiracy. According to unverified reports circulating across social media platforms, officials met behind closed doors to plan a secret initiative. Supporters of the claim argue that mainstream media outlets are intentionally suppressing the truth. Critics call it baseless speculation, but online forums continue to debate the mysterious revelations.",
        "Social media users are in an uproar following allegations of widespread wrongdoing. Viral posts claim that insider documents reveal a massive financial cover-up involving high-profile personalities. While independent fact-checkers have raised doubts about the authenticity of the documents, thousands of users continue to share the story across multiple platforms.",
        "In a dramatic turn of events, explosive rumor videos have sparked national debate. Commentators claim that recent policy changes were enacted without public consultation. Critics have demanded full transparency while supporters urge caution until official investigations yield concrete evidence.",
        "Sensational reports released today suggest an unprecedented scandal in government affairs. Source materials allegedly obtained from anonymous insiders hint at systematic manipulation of public records. Officials have refused to comment on the allegations, fueling further speculation online.",
        "A controversial new report claims that secret agreements were signed between major corporations and regulatory bodies. Although no official documentation has been produced to verify these statements, key opinion leaders argue that immediate oversight is necessary."
    ]

    real_texts = [
        "WASHINGTON (Reuters) - Lawmakers voted overwhelmingly on Tuesday to approve a multi-billion dollar infrastructure package aimed at upgrading national roads, bridges, and public transit systems. The legislation passed with bipartisan support following months of intense negotiations between congressional leaders. Administration officials praised the vote as a landmark achievement for federal infrastructure policy.",
        "LONDON (Reuters) - The central bank raised its benchmark interest rate by a quarter percentage point on Thursday in a continued effort to tame persistent inflationary pressures. Governor Andrew Miller noted that economic growth remains modest while labor markets show resilience. Financial markets reacted predictably to the announcement, with bond yields adjusting across key benchmark maturities.",
        "TOKYO (Reuters) - Leading technology companies announced a strategic alliance on Wednesday aimed at researching advanced semiconductor manufacturing processes. The joint venture will invest heavily in next-generation microchip design and supply chain diversification. Industry analysts welcomed the announcement as a key step toward mitigating global hardware shortages.",
        "PARIS (Reuters) - European heads of state gathered in Paris to review regional carbon reduction milestones under international environmental treaties. Delegates emphasized the importance of accelerating wind and solar energy deployment while maintaining grid stability. A joint communiqué issued at the conclusion of the summit outlined planned regulatory frameworks for 2030 target metrics.",
        "GENEVA (Reuters) - The World Health Organization released revised technical guidance today for seasonal respiratory disease surveillance. Public health experts emphasized standard immunization protocols and updated diagnostic testing methods for healthcare providers worldwide."
    ]

    fake_data = []
    real_data = []
    
    dates = ["December 31, 2017", "November 15, 2017", "October 20, 2017", "September 5, 2017", "August 12, 2017"]

    # Generate 1200 fake articles
    for i in range(1200):
        t_base = random.choice(fake_titles)
        b_base = random.choice(fake_texts)
        subj = random.choice(fake_subjects)
        dt = random.choice(dates)
        fake_data.append({
            "title": f"{t_base} (Report #{i+1})",
            "text": f"{b_base} Additional context regarding item {i+1}: mysterious developments were noted in regional forums where commentators debated the implications.",
            "subject": subj,
            "date": dt
        })

    # Generate 1200 real articles
    for i in range(1200):
        t_base = random.choice(real_titles)
        b_base = random.choice(real_texts)
        subj = random.choice(real_subjects)
        dt = random.choice(dates)
        real_data.append({
            "title": f"{t_base} - Update #{i+1}",
            "text": f"{b_base} Official statement {i+1} confirmed that parliamentary committees will monitor implementation timelines.",
            "subject": subj,
            "date": dt
        })

    pd.DataFrame(fake_data).to_csv(FAKE_PATH, index=False)
    pd.DataFrame(real_data).to_csv(TRUE_PATH, index=False)

    print(f"Saved {len(fake_data)} fake articles to {FAKE_PATH}")
    print(f"Saved {len(real_data)} real articles to {TRUE_PATH}")
