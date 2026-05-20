# S&P Sovereign Rating Audit Trail

This file documents the public sources used to compile each country's
year-end S&P Foreign-Currency Long-Term rating in `config/ratings.csv`.
The 14 `tests/test_ratings_sanity.py` cases verify the major transitions
mechanically; this document records the source-of-record for each.

## Sources of record

| Type | Source |
|---|---|
| Primary | S&P Global Ratings — Sovereign Default and Rating Transition Study (annual, published Q2). |
| Primary | S&P sovereign rating action press releases (`spglobal.com/ratings`). |
| Cross-check | Reuters and Bloomberg rating-action headlines via news archive. |
| Cross-check | Wikipedia "List of countries by credit rating", maintained from official S&P press releases. |

## Verified major transitions

The transitions below are public, dated, and widely cited.

| Country | Date | From | To | Direction | Note |
|---|---|---|---|---|---|
| BRA | 30 Apr 2008 | BB+ | BBB- | upgrade | First-ever investment grade for Brazil |
| BRA | 09 Sep 2015 | BBB- | BB+ | downgrade | Petrobras / fiscal credibility shock |
| IND | 30 Jan 2007 | BB+ | BBB- | upgrade | First IG for India |
| RUS | 14 Aug 1998 | B+ | CCC | downgrade | Sovereign default and rouble devaluation |
| RUS | 26 Apr 2014 | BBB | BBB- | downgrade | Crimea sanctions wave |
| RUS | 26 Jan 2015 | BBB- | BB+ | downgrade | Lost IG |
| RUS | 23 Feb 2018 | BB+ | BBB- | upgrade | Regained IG |
| RUS | 17 Mar 2022 | BB+ | CC | downgrade | Sanctions response to Ukraine invasion |
| RUS | 02 Apr 2022 | CC | SD | default | Foreign-currency selective default |
| ZAF | 25 Feb 2000 | BB+ | BBB- | upgrade | First IG since apartheid transition |
| ZAF | 03 Apr 2017 | BBB- | BB+ | downgrade | Cabinet reshuffle / fiscal slip |
| CHN | 21 Sep 2017 | AA- | A+ | downgrade | Rising leverage in shadow banking |
| MEX | 07 Feb 2002 | BB+ | BBB- | upgrade | First IG |
| IDN | 19 May 2017 | BB+ | BBB- | upgrade | First IG since AFC |
| PHL | 02 May 2013 | BB+ | BBB- | upgrade | First IG |
| PER | 14 Jul 2008 | BB+ | BBB- | upgrade | First IG |
| COL | 19 May 2021 | BBB- | BB+ | downgrade | Lost IG, fiscal pressure |
| MAR | 23 Mar 2010 | BB+ | BBB- | upgrade | First IG |
| MAR | 02 Apr 2021 | BBB- | BB+ | downgrade | Lost IG |

## Always-AAA

Singapore has been AAA on S&P's main scale since the 6 Mar 1995 upgrade
and has not moved since.

## Unrated by S&P

S&P does not publish ratings for the AES Sahel countries (Mali, Niger)
at any point in the 1995–2024 window. Burkina Faso entered S&P's
perimeter in 2004 at B/Stable; Rwanda entered in June 2009 at B/Stable;
Kenya entered in 2006 at B+/Stable.

## Lowest-confidence cells

The following cells in `config/ratings.csv` are the ones the author
least confidently nails to a public press release. Before journal
submission, re-verify each against S&P's per-country rating-history
page (`spglobal.com/ratings/en/research-insights/sovereign-credit-ratings`).

- BFA 1995–2003 (NR; first rated in 2004)
- BFA 2015–2017 transitions (B → B- → B)
- RWA 2009 initial rating (compiled as B, S&P first action June 2009)
- KEN 2006 initial rating (B+, first S&P action 2006)
- PAK 1998 sanction-related downgrade (compiled as B; some sources have B-)
- PAK 1999 default rating (compiled as CCC; precise notch at point of default varies)

## Pytest sanity tests

`tests/test_ratings_sanity.py` verifies 14 transition sets mechanically.
All 14 currently pass and gate the build. They cover Brazil 2008/2015,
India 2007, Russia 2008/2014/2018/2022, South Africa 2000/2017,
China 2017, Morocco 2010/2021, Mexico 2002, Indonesia 2017,
Philippines 2013, Peru 2008, Colombia 2021, Pakistan 2022, AES
Sahel unrated, Singapore AAA-continuous.
