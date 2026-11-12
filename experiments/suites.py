"""Public datasets turned into decision-head suites with separate train and test cases.

Each builder returns {"name", "questions": {key: question}, "train": [case], "test": [case]},
where a case is {"id", "state", "expected": {key: answer}} as in cases/*.json.

    clinc_<domain>   choice: one CLINC150 domain (3 used for tuning, 7 held out) (15 intents) plus "out_of_scope", which is
                     trained and tested on CLINC's labelled out-of-scope queries
    banking77_g<k>   choice: one of four fixed groups of ~20 Banking77 intents (77 options do
                     not fit the ~34 single-token handles)
    sst5             score: SST-5 movie-review sentiment, levels 0 (very negative) to 4

Option descriptions are just the intent names in words ("bill_due" -> "bill due"), so the
zero-data numbers are a floor that better-written descriptions would raise.
"""

import random
from collections import defaultdict

MAX_TRAIN_PER_OPTION = 100  # the largest learning-curve size; more is never used
SEED = 0  # fixes which examples are sampled, so cached features stay valid

# Domains as grouped in the CLINC150 paper (Larson et al., 2019); the Hugging Face copy of the
# dataset has intents only.
CLINC_DOMAINS = {
    "banking": "transfer transactions balance freeze_account pay_bill bill_balance bill_due "
               "interest_rate routing min_payment order_checks pin_change report_fraud "
               "account_blocked spending_history",
    "credit_cards": "credit_score report_lost_card credit_limit rewards_balance new_card "
                    "application_status card_declined international_fees apr redeem_rewards "
                    "credit_limit_change damaged_card replacement_card_duration "
                    "improve_credit_score expiration_date",
    "travel": "book_flight book_hotel car_rental travel_suggestion travel_alert "
              "travel_notification carry_on timezone vaccines translate flight_status "
              "international_visa lost_luggage plug_type exchange_rate",
}
# The other seven domains were held out: no training setting was chosen by looking at them.
# Run them once, with the settings frozen, to check those settings were not fitted to the suites
# above (python -m experiments.learning_curves --holdout).
CLINC_HOLDOUT_DOMAINS = {
    "kitchen_and_dining": "recipe food_last meal_suggestion nutrition_info ingredient_substitution "
                          "calories cook_time restaurant_reservation restaurant_reviews "
                          "accept_reservations confirm_reservation cancel_reservation how_busy "
                          "ingredients_list restaurant_suggestion",
    "home": "what_song play_music todo_list_update reminder reminder_update calendar_update "
            "order_status update_playlist shopping_list calendar next_song order todo_list "
            "shopping_list_update smart_home",
    "auto_and_commute": "traffic directions gas gas_type distance current_location mpg "
                        "oil_change_when oil_change_how jump_start uber schedule_maintenance "
                        "last_maintenance tire_pressure tire_change",
    "utility": "weather alarm date find_phone share_location timer make_call calculator definition "
               "measurement_conversion flip_coin spelling time roll_dice text",
    "work": "pto_request pto_request_status next_holiday insurance insurance_change pto_balance "
            "pto_used w2 schedule_meeting meeting_schedule rollover_401k income taxes "
            "direct_deposit payday",
    "small_talk": "who_made_you meaning_of_life who_do_you_work_for do_you_have_pets "
                  "what_are_your_hobbies fun_fact what_is_your_name where_are_you_from goodbye "
                  "thank_you greeting tell_joke are_you_a_bot how_old_are_you what_can_i_ask_you",
    "meta": "change_speed user_name whisper_mode yes change_volume no change_language repeat "
            "change_accent cancel sync_device change_user_name change_ai_name reset_settings maybe",
}
CLINC_OOS_TRAIN = 100  # out-of-scope training examples (as many as each intent has)
CLINC_OOS_TEST = 150   # of CLINC's 1,000 out-of-scope test queries, sampled
BANKING77_GROUPS = 4
SST5_TEST = 500        # sampled from the 2,210 test reviews, keeping the natural label mix


def _human(name):
    return name.replace("_", " ").rstrip("?").strip()


def _load(name, config=None):
    from datasets import load_dataset

    return load_dataset(name, config) if config else load_dataset(name)


def _cap_per_label(rows, per_label, rng):
    """Shuffle (text, label) rows and keep at most `per_label` of each label."""
    rows = rows[:]
    rng.shuffle(rows)
    seen, out = defaultdict(int), []
    for text, label in rows:
        if seen[label] < per_label:
            seen[label] += 1
            out.append((text, label))
    return out


def _shuffled(rows, rng):
    rows = rows[:]
    rng.shuffle(rows)
    return rows


def _cases(rows, key, split):
    return [{"id": f"{split}{i}", "state": text, "expected": {key: label}}
            for i, (text, label) in enumerate(rows)]


def _suite(name, key, question, train, test):
    return {"name": name, "questions": {key: question},
            "train": _cases(train, key, "train"), "test": _cases(test, key, "test")}


def clinc(domain):
    ds = _load("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    intents = {**CLINC_DOMAINS, **CLINC_HOLDOUT_DOMAINS}[domain].split()
    criteria = {i: _human(i) for i in intents}
    criteria["out_of_scope"] = "none of the above: the request is about something else"
    question = {"type": "choice",
                "instructions": f"Which {_human(domain)} request is this",
                "criteria": criteria}
    rng = random.Random(SEED)

    def rows(split, n_oos):
        r = [(x["text"], names[x["intent"]]) for x in ds[split]]
        inside = [(t, lab) for t, lab in r if lab in criteria]
        oos = [(t, "out_of_scope") for t, lab in r if lab == "oos"]
        rng.shuffle(oos)
        return inside + oos[:n_oos]

    train = _cap_per_label(rows("train", CLINC_OOS_TRAIN), MAX_TRAIN_PER_OPTION, rng)
    test = _shuffled(rows("test", CLINC_OOS_TEST), rng)  # mixed, so --limit slices are too
    return _suite(f"clinc_{domain}", "intent", question, train, test)


def banking77(group):
    ds = _load("PolyAI/banking77")
    names = ds["train"].features["label"].names
    order = list(range(len(names)))
    random.Random(SEED).shuffle(order)
    labels = [names[i] for i in order[group::BANKING77_GROUPS]]
    question = {"type": "choice",
                "instructions": "Which banking support topic is this customer message about",
                "criteria": {lab: _human(lab) for lab in labels}}
    keep = set(labels)
    rng = random.Random(SEED)

    def rows(split):
        return [(x["text"], names[x["label"]]) for x in ds[split] if names[x["label"]] in keep]

    train = _cap_per_label(rows("train"), MAX_TRAIN_PER_OPTION, rng)
    return _suite(f"banking77_g{group}", "intent", question, train, _shuffled(rows("test"), rng))


def sst5():
    ds = _load("SetFit/sst5")
    question = {"type": "score",
                "instructions": "How positive the sentiment of this movie review is",
                "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}
    rng = random.Random(SEED)
    train = _cap_per_label([(x["text"], x["label"]) for x in ds["train"]], MAX_TRAIN_PER_OPTION, rng)
    test = [(x["text"], x["label"]) for x in ds["test"]]
    rng.shuffle(test)
    return _suite("sst5", "sentiment", question, train, test[:SST5_TEST])


def build(name):
    if name.startswith("clinc_"):
        return clinc(name[len("clinc_"):])
    if name.startswith("banking77_g"):
        return banking77(int(name[len("banking77_g"):]))
    if name == "sst5":
        return sst5()
    raise ValueError(f"unknown suite {name!r}")


ALL = [f"clinc_{d}" for d in CLINC_DOMAINS] + [f"banking77_g{g}" for g in range(BANKING77_GROUPS)] + ["sst5"]
HOLDOUT = [f"clinc_{d}" for d in CLINC_HOLDOUT_DOMAINS]
