"""Curated content pools: the hidden truths the game is built from.

The truth of a case is SAMPLED IN CODE from these pools (REBUILD_REVIEW.md §10.2/§13.4):
each profession carries a domain tag and 3 domain-matched faults, so profession/fault
coherence and the smokescreen (profession-domain ⊥ jargon-style) are guarantees, never
learned behaviors. The director model only ever writes the oblique facts.

Pure Python, no dependencies — imported by both the runtime (buzzwords.pipeline) and the
offline training scripts on Modal (mounted with add_local_python_source).

Fault strings are VERB PHRASES completing "The defendant ___".
"""

from __future__ import annotations

import random

# Jargon style -> profession domains that would break the smokescreen if sampled.
STYLE_EXCLUDED_DOMAINS = {
    "corporate": {"business"},
    "aviation": {"aviation"},
    "ai": {"tech"},
    "politics": {"politics"},
    "medical": {"medical"},
    "gaming": {"gaming"},
    "sports": {"sports"},
    "scifi": {"space", "tech"},
}

# profession -> (domain, [3 matched faults])
PROFESSIONS: dict[str, tuple[str, list[str]]] = {
    # ---- food ----
    "pastry chef": ("food", [
        "served a wedding cake he knew had spoiled rather than refund the order",
        "passed off a supplier's frozen pastries as his own handmade work",
        "kept using a cracked oven the inspector had already condemned",
    ]),
    "sushi chef": ("food", [
        "served farmed fish at wild-caught prices for over a year",
        "broke the cold chain on a delivery and served it anyway",
        "let an untrained cousin prepare blowfish for paying customers",
    ]),
    "food-truck owner": ("food", [
        "ran the grill for a month after the gas line failed inspection",
        "reused single-use gloves and oil far past every safety rule",
        "parked in a rival's licensed spot and traded under their permit number",
    ]),
    "sommelier": ("food", [
        "refilled grand-cru bottles with table wine and resealed them",
        "invented tasting notes for a cellar he had never opened",
        "sold a client's stored collection and reported it as spoilage",
    ]),
    "butcher": ("food", [
        "relabeled expired cuts with fresh dates every Monday",
        "sold ordinary beef as certified organic at double the price",
        "ground yesterday's unsold stock into the 'fresh daily' mince",
    ]),
    "barista": ("food", [
        "served decaf as espresso all winter to stretch inventory",
        "scraped mold off syrup pumps instead of replacing them",
        "pocketed the tip jar and blamed the closing shift",
    ]),
    # ---- arts & events ----
    "wedding photographer": ("arts", [
        "deleted the only copies of a ceremony she was paid to cover and blamed the gear",
        "resold a couple's private photos to a stock-image site",
        "double-booked two ceremonies and sent an unpaid intern to one",
    ]),
    "museum curator": ("arts", [
        "falsified a provenance record to cover a quiet theft",
        "swapped a genuine exhibit for a replica and sold the original",
        "back-dated a loan agreement to hide that a piece had left the building",
    ]),
    "tattoo artist": ("arts", [
        "reused needles on walk-in clients to save on supplies",
        "inked a copyrighted design he had sworn was his own",
        "worked on a sedated client who had never signed the consent form",
    ]),
    "portrait painter": ("arts", [
        "sold the same 'one of a kind' commission to three different buyers",
        "had a student produce the commissions he signed and delivered",
        "used a client's sittings to copy and sell her likeness without consent",
    ]),
    "wedding DJ": ("arts", [
        "took deposits for a summer of receptions he never intended to play",
        "ran unlicensed tracks at corporate events and pocketed the licensing fee",
        "left a reception mid-set for a better-paying gig across town",
    ]),
    "stage manager": ("arts", [
        "signed off on a rigging check that was never performed",
        "let an actor go on under rig lights flagged as a fire risk",
        "forged the venue's safety certificate to keep opening night",
    ]),
    "antiques dealer": ("arts", [
        "aged reproduction furniture and sold it as period originals",
        "bought from house clearances at scrap prices by lying about value",
        "laundered a stolen estate collection through his shop ledger",
    ]),
    # ---- trades ----
    "plumber": ("trades", [
        "signed off on work an unlicensed trainee had actually done",
        "billed three callouts for a leak he caused on the first visit",
        "installed second-hand parts and invoiced them as new",
    ]),
    "locksmith": ("trades", [
        "kept copies of a client's keys and let himself in uninvited",
        "sold a building's master-key pattern to a burglary crew",
        "charged for high-security locks and fitted the cheap line instead",
    ]),
    "electrician": ("trades", [
        "skipped the earth-bond test on a nursery he certified as safe",
        "buried a known faulty junction behind a finished wall",
        "copied another firm's certificate stamp to pass his own work",
    ]),
    "roofer": ("trades", [
        "took storm-repair deposits from a whole street and vanished",
        "patched over rot he was paid to replace",
        "stripped lead from one job to sell and billed the next job for it",
    ]),
    "elevator technician": ("trades", [
        "stretched mandatory monthly inspections to twice a year and back-filled the log",
        "cleared a fault code without fixing the brake that threw it",
        "let a building run on an expired safety certificate he promised to renew",
    ]),
    "auto mechanic": ("trades", [
        "rolled back odometers on trade-ins for a dealer friend",
        "charged for brake pads he never fitted",
        "passed a failed inspection in exchange for cash",
    ]),
    "house painter": ("trades", [
        "thinned premium paint and billed for twice the cans",
        "painted over damp the surveyor had flagged for treatment",
        "used the client's empty property for a week of parties between coats",
    ]),
    # ---- nature & animals ----
    "marine biologist": ("nature", [
        "released unverified findings a rival lab later had to retract",
        "back-dated tide samples to fit a grant deadline",
        "credited a junior's discovery to herself, then removed her name entirely",
    ]),
    "beekeeper": ("nature", [
        "cut premium honey with corn syrup and kept the organic label",
        "moved infected hives into a neighbor's orchard at night",
        "sold colonies he knew carried mites to first-time keepers",
    ]),
    "park ranger": ("nature", [
        "sold the locations of protected nests to trophy collectors",
        "logged patrols of trails he had not walked in months",
        "let a poaching ring operate in exchange for a cut",
    ]),
    "florist": ("nature", [
        "swapped in cheaper stock and pocketed the difference on a big order",
        "resold a funeral's arrangements to a wedding the same afternoon",
        "imported restricted plants hidden inside bouquet shipments",
    ]),
    "dog groomer": ("nature", [
        "sedated difficult dogs without consent or a license",
        "kept clients' show dogs overnight to breed from them in secret",
        "covered up an injury a clipper caused and blamed the owner",
    ]),
    "landscape gardener": ("nature", [
        "billed a council for trees that were never planted",
        "dumped clients' garden waste in a protected wetland",
        "cut a neighbor's century oak because it shaded his client's lawn",
    ]),
    # ---- transport (ground/sea) ----
    "city bus driver": ("transport", [
        "skipped the morning brake walk-around for months to leave the depot early",
        "kept driving a route after his medical certificate lapsed",
        "logged rest breaks he spent doing a second courier job",
    ]),
    "ferry captain": ("transport", [
        "sailed overloaded on holiday weekends and falsified the passenger count",
        "skipped the lifeboat drill all season and signed the drill log anyway",
        "let an unlicensed mate take the helm through the narrows",
    ]),
    "crane operator": ("transport", [
        "lifted loads over an occupied school after the exclusion order",
        "ignored the wind-speed cutoff to finish a pour before the weekend",
        "let his certification lapse and traded shifts to dodge the assessor",
    ]),
    "delivery courier": ("transport", [
        "marked parcels delivered and sold the contents online",
        "forged signatures for an entire apartment block for a year",
        "dumped a vanload of 'next-day' parcels in a storage unit each Friday",
    ]),
    "driving instructor": ("transport", [
        "passed students who failed in exchange for cash",
        "logged lessons that never happened against prepaid packages",
        "taught for two years on a suspended license of his own",
    ]),
    "tow-truck driver": ("transport", [
        "staged breakdowns with a garage to split inflated repair bills",
        "towed legally parked cars from a lot he had a kickback deal with",
        "stripped parts from impounded cars and reported them as missing on arrival",
    ]),
    # ---- education ----
    "chemistry teacher": ("education", [
        "negligently skipped the fume-hood check before a class demonstration",
        "let students handle reagents the district had banned from classrooms",
        "inflated lab grades for athletes at the principal's quiet request",
    ]),
    "kindergarten teacher": ("education", [
        "left the class with the janitor to run personal errands",
        "signed the daily headcount without doing it, the day a child wandered off",
        "spent the class trip fund on herself and cancelled the trip",
    ]),
    "librarian": ("education", [
        "sold rare first editions from the archive and shelved facsimiles",
        "doctored the catalogue to hide a decade of missing volumes",
        "leaked patrons' borrowing records to a private investigator",
    ]),
    "piano teacher": ("education", [
        "charged conservatory-prep rates while inventing his own credentials",
        "entered students in competitions and kept the prize money",
        "sold a prodigy's audition recordings without the family's consent",
    ]),
    # ---- retail & service ----
    "hotel concierge": ("service", [
        "sold guests' room numbers and schedules to paparazzi",
        "ran a scalping ring out of the theater-desk allocation",
        "copied room keys for an after-hours luggage-theft crew",
    ]),
    "hairdresser": ("service", [
        "used an unlicensed chemical straightener that burned a client's scalp",
        "sold clients' cut hair to a wig maker without telling them",
        "kept booking patch-test treatments with no patch tests for years",
    ]),
    "dry cleaner": ("service", [
        "blamed 'lost' garments on customers while reselling them at a stall",
        "used a banned solvent and vented it into the alley behind a café",
        "swapped designer buttons and trims for replicas before returning suits",
    ]),
    "pawnbroker": ("service", [
        "melted down flagged stolen jewelry before the police check cleared",
        "lent against items he then reported stolen for the insurance",
        "rewrote ticket dates so forfeits landed a week early",
    ]),
    "wedding planner": ("service", [
        "took kickbacks from vendors and billed couples the full rate",
        "spent one couple's deposit to cover another couple's overruns for years",
        "booked a venue she knew was double-reserved and gambled on a cancellation",
    ]),
    "real-estate agent": ("service", [
        "staged fake rival bids to drive up closing prices",
        "sold a house without disclosing the subsidence report he had read",
        "rented out a seller's empty property for cash while it was listed",
    ]),
    # ---- style-domain professions (excluded for their matching jargon) ----
    "airline pilot": ("aviation", [
        "skipped a mandatory pre-flight checklist item to make a slot time",
        "flew the return leg knowing a warning light had been deferred improperly",
        "under-reported a hard landing to dodge the inspection it required",
    ]),
    "air-traffic controller": ("aviation", [
        "worked a double shift unrested and off the books during a staffing audit",
        "let a friend's charter jump the queue ahead of holding aircraft",
        "deleted a near-miss tape instead of filing the report",
    ]),
    "flight attendant": ("aviation", [
        "skipped the door-arming cross-check she signed for",
        "smuggled duty-free stock past customs in the crew bag for resale",
        "served a visibly drunk passenger who later opened a galley exit lever",
    ]),
    "aircraft mechanic": ("aviation", [
        "signed off a torque check he never performed",
        "fitted an uncertified part and copied the serial from a scrapped one",
        "pencil-whipped a corrosion inspection to clear the morning departure",
    ]),
    "accountant": ("business", [
        "moved client money through a personal account to 'smooth' month-end",
        "shredded invoices ahead of an audit he knew was coming",
        "kept two ledgers for a restaurant client for five years",
    ]),
    "bank teller": ("business", [
        "skimmed dormant accounts a few cents at a time for years",
        "tipped off a friend about a customer's withdrawal schedule",
        "processed a forged signature she recognized and said nothing",
    ]),
    "insurance adjuster": ("business", [
        "inflated estimates and split the difference with a contractor",
        "denied valid claims to hit a quarterly target",
        "back-dated a policy for a relative after the storm hit",
    ]),
    "auctioneer": ("business", [
        "planted shill bidders in the room at his own sales",
        "knocked lots down early to a partner before reserve was met",
        "sold a consignor's lot privately and reported it unsold",
    ]),
    "pharmacist": ("medical", [
        "diluted high-cost prescriptions and billed for the full dose",
        "filled prescriptions she knew were forged for a regular",
        "swapped brand-name pills for expired generics and kept the difference",
    ]),
    "dental hygienist": ("medical", [
        "reused single-use instruments between patients to cut costs",
        "performed procedures only the dentist was licensed to do",
        "upsold whitening treatments using scans of other people's teeth",
    ]),
    "paramedic": ("medical", [
        "logged response times minutes earlier than the truth all year",
        "pocketed controlled painkillers and recorded them as administered",
        "skipped the equipment check the morning the defibrillator failed",
    ]),
    "optometrist": ("medical", [
        "prescribed lenses patients did not need to hit a sales quota",
        "signed driving-vision certificates without doing the field test",
        "billed insurers for exams on patients who never visited",
    ]),
    "app developer": ("tech", [
        "shipped a paid update he knew silently drained customer batteries",
        "copied a rival's codebase and re-skinned it as his own product",
        "sold user location data despite the 'no tracking' promise on the store page",
    ]),
    "IT support technician": ("tech", [
        "browsed and copied executives' private files while 'fixing' laptops",
        "installed remote-access tools on clients' machines for later use",
        "billed for antivirus licenses he never activated",
    ]),
    "data-center technician": ("tech", [
        "mined cryptocurrency on customers' idle servers for a year",
        "faked the generator load-test logs the week the power failed",
        "resold decommissioned drives without wiping them",
    ]),
    "campaign manager": ("politics", [
        "paid for fake local endorsements out of petty cash",
        "coordinated illegally with an outside spending group",
        "buried an internal poll and reported the donor-friendly one",
    ]),
    "city council clerk": ("politics", [
        "back-dated meeting minutes to legalize a vote that never happened",
        "leaked sealed bid amounts to a favored contractor",
        "shredded public-records requests instead of logging them",
    ]),
    "esports coach": ("gaming", [
        "made minors scrim ten-hour days off the books against league rules",
        "sold his own team's strategies to a rival organization",
        "pressured a player to throw a qualifier for a betting syndicate",
    ]),
    "arcade owner": ("gaming", [
        "rigged the claw machines below the legal win rate",
        "paid prize tickets out at half value to kids who couldn't do the math",
        "ran unlicensed gambling cabinets in the back room",
    ]),
    "gym instructor": ("sports", [
        "trained clients for a year on a first-aid certificate he forged",
        "sold banned supplements to teenage members from his locker",
        "kept billing cancelled memberships and pocketed the 'errors'",
    ]),
    "football referee": ("sports", [
        "fixed bookings in lower-league matches for a betting ring",
        "passed fitness tests by sending a lookalike cousin",
        "tipped team sheets to gamblers an hour before kickoff",
    ]),
    "ski instructor": ("sports", [
        "took beginners off-piste in closed avalanche conditions to pad lesson fees",
        "taught all season with an expired mountain-safety certification",
        "filed a fake injury claim against the resort with a friend as witness",
    ]),
    "planetarium presenter": ("space", [
        "pocketed school-group entry fees by running unticketed shows",
        "sold 'name a star' certificates for a registry that never existed",
        "let the dome's fire-exit inspection lapse and forged the sticker",
    ]),
    "observatory technician": ("space", [
        "sold reserved telescope time to private clients off the schedule",
        "fabricated calibration logs after dropping a survey mirror",
        "leaked embargoed survey data to a collector of asteroid claims",
    ]),
}

# Narrative seasoning for the teacher's transcripts (training-side variety axes).
TONES = ["combative and theatrical", "dry and procedural", "indignant and moralizing",
         "coldly clinical", "exasperated and impatient"]
DISPOSITIONS = ["defiant", "remorseful", "oblivious and confused", "smug and evasive"]

# Last-resort oblique facts if the director model fails to produce valid ones at
# runtime (generic enough to fit any sampled case; the game stays playable).
FALLBACK_FACTS = [
    "the paperwork tells a different story than the testimony",
    "a routine record shows a gap exactly where it matters",
    "someone double-checked the work, and what they found did not match",
    "the people who paid for the service were the last to learn the truth",
]


def eligible_professions(style: str) -> list[str]:
    """Professions whose domain does not collide with the jargon style (smokescreen)."""
    excluded = STYLE_EXCLUDED_DOMAINS.get(style, set())
    return [p for p, (domain, _) in PROFESSIONS.items() if domain not in excluded]


def sample_case(rng: random.Random, style: str) -> tuple[str, str]:
    """Sample (profession, matched fault) for a jargon style. Smokescreen + profession/
    fault coherence hold by construction (REBUILD_REVIEW.md §13.4)."""
    profession = rng.choice(eligible_professions(style))
    return profession, rng.choice(PROFESSIONS[profession][1])


def banned_words(profession: str) -> list[str]:
    """Profession tokens that must never appear in facts/stage directions/lines."""
    tokens = [profession] + [w for w in profession.replace("-", " ").split() if len(w) > 3]
    seen, out = set(), []
    for t in tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out

