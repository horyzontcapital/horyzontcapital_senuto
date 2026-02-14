#!/usr/bin/env python3
"""
Senuto API - Runda 1 executor

Builds the list of required API calls from the bundled OpenAPI spec and runs
them for all domains listed in data/wszystkie_domeny.txt. The flow mirrors the
instructions in zadanie.txt (Round 1 = 10 endpoints per domain) and the
validated behaviour from senuto.py (all calls work over GET with query params).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "api_specs.json"
DOMAINS_PATH = ROOT / "data" / "wszystkie_domeny.txt"
OUTPUT_DIR = ROOT / "data" / "runda2_outputs"
PARTIAL_DIR = OUTPUT_DIR / "partial"

API_BASE = os.getenv("SENUTO_API_BASE", "https://api.senuto.com")
TOKEN = os.getenv(
	"SENUTO_TOKEN",
	"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MTg2Mzg3LCJleHAiOjE3NzM1MjcwOTIsImlhdCI6MTc3MDg0ODY5Mn0.GJR6hIyp46jtK48JKfACCx7vsLAS6XZM6Njo9x5Nq2M",
)
COUNTRY_ID = os.getenv("SENUTO_COUNTRY_ID", "200")
FETCH_MODE = os.getenv("SENUTO_FETCH_MODE", "topLevelDomain")
DELAY = float(os.getenv("SENUTO_DELAY", "1.5"))

CHARACTERISTICS = [
	"words_count",
	"trends",
	"searches",
	"difficulty",
	"keyword_params",
	"serp_params",
]

# ----------------------------------------------------------------------------
# DOMAIN GROUPING
# ----------------------------------------------------------------------------

# Domain groups with pagination limits
# Group A (direct competition) + Group B (fintech) + HC = unlimited pages
# Group C (comparison portals) = max 10 pages for positions_data and cannibalization_keywords

DOMAIN_GROUPS = {
	"A": {  # Bezpo[rednia konkurencja - 48 domen (27 old + 21 new)
		# Istniejce (27):
		"baltcredit.pl", "novalend.pl", "pozyczkihipoteczne.eu", "yanokhipoteczny.pl",
		"capitalaccess.pl", "businessfunds.pl", "forsel.pl", "profikredyt.com.pl",
		"topf.pl", "krk-partners.pl", "trendcapital.pl", "janvest.pl",
		"e-bilans.info", "funduszkorona.pl", "ibipolska.pl", "tadam-finanse.pl",
		"actius.pl", "alphacredit.pl", "credithub.pl", "pod-zastaw.org",
		"pozyczka-pod-zastaw.com", "cpf-pozyczki.pl", "bankowybroker.pl",
		"mldomfinansowy.pl", "moneyrun.pl", "khfinanse.com", "bankierkredyty.eu",
		# Nowe Runda 1.5 (21):
		"podzastaw.biz", "gotowka-podzastaw.letnet.pl", "smartvest.pl",
		"inwestum.com.pl", "habzafinanse.com.pl", "newpoint-finance.pl",
		"fluxo.com.pl", "finiata.pl", "serwaczakowie.pl", "cbif.pl",
		"ftfund.pl", "ckakrybia.pl", "firmowefinanse.pl", "hipotecznapozyczka.com",
		"pozyczkipodzastaw.com.pl", "simplefinance.pl", "mevior.pl",
		"goldenexpert.pl", "gawryszewskifinanse.pl", "splatapozyczek.pl", "open.pl",
	},
	"B": {  # Fintech - 7 domen
		"smeo.pl", "finea.pl", "pragmago.pl", "wealthon.com",
		"bizlend.pl", "smartney.pl", "pl.finbee.com",
	},
	"C": {  # Porównywarki - 47 domen (37 old + 10 new)
		# Istniejce (37):
		"bankujesz.pl", "czerwona-skarbonka.pl", "comperia.pl", "loando.pl",
		"bankier.pl", "totalmoney.pl", "wnioskomat.com", "money.pl",
		"odpowiedzialnybiznes.pl", "kontomaniak.pl", "lendi.pl", "expander.pl",
		"rankomat.pl", "moneteo.com", "mfinanse.pl", "rekinfinansow.pl",
		"helikon.net.pl", "feniksfinanse.pl", "bancovo.pl", "fintek.pl",
		"notus.pl", "confronter.pl", "credipass.pl", "finergia.pl",
		"grupasg.pl", "kredytomat.pl", "platinum-financial.pl", "monevia.pl",
		"dginwest.com.pl", "brsonline.pl", "hiponet.pl", "strefabiznesu.pl",
		"poradnikprzedsiebiorcy.pl", "fines.pl", "businessinsider.com.pl",
		"janosikdlafirm.pl", "exfinder.pl",
		# Nowe Runda 1.5 (10):
		"kasastefczyka.pl", "forum-kredytowe.pl", "blogkredytowy.pl",
		"angfinanse.pl", "kokos.pl", "akredo.pl", "finhack.pl",
		"bankoweabc.pl", "smart-money.pl", "helpfind.pl",
	},
	"HC": {"horyzontcapital.com"},
}

# Endpoints with Group C page limits
GROUP_C_LIMITED_ENDPOINTS = ["get_positions_data", "get_cannibalization_keywords"]
GROUP_C_MAX_PAGES = 10

def get_domain_group(domain: str) -> str:
	"""Return group (A, B, C, HC) for a domain."""
	clean = clean_domain(domain)
	for group_name, domains in DOMAIN_GROUPS.items():
		if clean in domains:
			return group_name
	return "A"  # Unknown domains default to Group A (unlimited)

# ----------------------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------------------

# Endpoint seeds (filled with spec metadata at runtime)
ENDPOINT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "get_domain_statistics",
        "spec_path": "/api/visibility_analysis/reports/dashboard/getDomainStatistics",
        "method": "get",
        "desc": "Widocznosc, TOP3/10/50, domain rank",
        "paged": False,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
        },
    },
    {
        "name": "get_positions_data",
        "spec_path": "/api/visibility_analysis/reports/positions/getData",
        "method": "post",
        "desc": "Frazy + pozycje + volume + CPC + trudnosc",
        "paged": True,
        "limit": 100,
        "max_pages": 50,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
			"page": 1,
			"limit": 100,
			"order": {"dir": "desc", "prop": "statistics.visibility.diff"},
			"isDataReadyToLoad": True,
        },
    },
    {
        "name": "get_competitors",
        "spec_path": "/api/visibility_analysis/reports/competitors/getData",
        "method": "post",
        "desc": "Auto-detect konkurentow",
        "paged": True,
        "limit": 100,
        "max_pages": 10,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
			"page": 1,
			"limit": 100,
			"order": {"dir": "desc", "prop": "statistics.visibility.diff"},
			"days_compare_mode": "last_monday",
        },
    },
    {
        "name": "get_urls",
        "spec_path": "/api/visibility_analysis/reports/sections/getUrls",
        "method": "post",
        "desc": "Lista URL-i z widocznoscia (sections)",
        "paged": True,
		"limit": 100,
        "max_pages": 50,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
			"order": {"dir": "desc", "prop": "statistics.visibility.current"},
        },
    },
    {
        "name": "get_url_sections",
        "spec_path": "/api/visibility_analysis/reports/sections/getSections",
        "method": "post",
        "desc": "Widocznosc per sekcja URL",
        "paged": True,
        "limit": 100,
        "max_pages": 50,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
			"order": {"dir": "desc", "prop": "statistics.visibility.current"},
        },
        # fetch_mode: `topLevelDomain`, `subdomain`
    },
    {
        "name": "get_positions_history_chart",
        "spec_path": "/api/visibility_analysis/reports/domain_positions/getPositionsHistoryChartDataForAllTypes",
        "method": "get",
        "desc": "Historia widocznosci w czasie",
        "paged": False,
        "params": lambda d: {
            "page": 1,
            "limit": 25,
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
            "date_min": "2024-01-01",
            "date_max": "2025-12-31",
            "date_interval": "weekly",
        },
        # date_interval: `weekly`, `daily`
    },
    {
        "name": "get_cannibalization_keywords",
        "spec_path": "/api/visibility_analysis/reports/cannibalization/getKeywords",
        "method": "post",
        "desc": "Kanibalizacja fraz",
		"paged": True,
		"limit": 100,
		"max_pages": 20,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
			"page": 1,
			"limit": 100,
        },
        # fetch_mode: `topLevelDomain`, `subdomain`
    },
    {
        "name": "get_subdomains",
        "spec_path": "/api/visibility_analysis/reports/sections/getSubdomains",
        "method": "post",
        "desc": "Analiza subdomen (sections)",
        "paged": True,
        "limit": 100,
        "max_pages": 5,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": "subdomain",
            "country_id": COUNTRY_ID,
			"page": 1,
			"limit": 100,
			"order": {"dir": "desc", "prop": "statistics.visibility.current"},
        },
        # fetch_mode: `topLevelDomain`, `subdomain`
    },
    {
        "name": "get_characteristics_table",
        "spec_path": "/api/visibility_analysis/reports/keywords/getCharacteristicsTable",
        "method": "post",
        "desc": "Segmentacja fraz (6 typow)",
        "paged": False,
        "multi_char": True,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
            "characteristics": "words_count",
        },
        # characteristics: `words_count`, `trends`, `searches`, `difficulty`, `keyword_params`, `serp_params`
        # days_compare_mode: `week_ago_monday`, `last_monday`, `yesterday`
    },
    {
        "name": "suggest_domains",
        "spec_path": "/api/visibility_analysis/domains_suggester/suggest",
        "method": "get",
        "desc": "Sugestie podobnych domen",
        "paged": False,
        "params": lambda d: {"value": stem_domain(d), "country_id": COUNTRY_ID},
    },

    # === RUNDA 1.5 - NOWE ENDPOINTY ===

    {
        "name": "get_positions_wins",
        "spec_path": "/api/visibility_analysis/reports/positions/getWins",
        "method": "post",
        "desc": "Frazy gained (wzrost pozycji)",
        "paged": True,
        "limit": 100,
        "max_pages": 50,
        "params": lambda d: {
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "country_id": COUNTRY_ID,
            "page": 1,
            "limit": 100,
            "days_compare_mode": "yesterday",
        },
    },
    {
        "name": "get_history_keywords_wins",
        "spec_path": "/api/visibility_analysis/reports/history/keywords/getWins",
        "method": "post",
        "desc": "Historia keywords gained (WYMAGA date_min/max)",
        "paged": True,
        "limit": 100,
        "max_pages": 20,
        "params": lambda d: {
            "country_id": COUNTRY_ID,
            "date_min": "2026-02-01",  # 1. dzieD obecnego miesica
            "date_max": "2026-02-13",  # Dzisiejsza data
            "days_compare_mode": "yesterday",
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "isDataReadyToLoad": True,
            "limit": 100,
            "order": {"dir": "desc", "prop": "statistics.visibility.diff"},
            "page": 1,
        },
    },
    {
        "name": "get_history_keywords_losses",
        "spec_path": "/api/visibility_analysis/reports/history/keywords/getLosses",
        "method": "post",
        "desc": "Historia keywords lost (WYMAGA date_min/max)",
        "paged": True,
        "limit": 100,
        "max_pages": 20,
        "params": lambda d: {
            "country_id": COUNTRY_ID,
            "date_min": "2026-02-01",
            "date_max": "2026-02-13",
            "days_compare_mode": "yesterday",
            "domain": d,
            "fetch_mode": FETCH_MODE,
            "isDataReadyToLoad": True,
            "limit": 100,
            "order": {"dir": "desc", "prop": "statistics.visibility.diff"},
            "page": 1,
        },
    },
    {
        "name": "get_direct_answers",
        "spec_path": "/api/visibility_analysis/tools/direct_answers/getData",
        "method": "post",
        "desc": "Featured snippets / direct answers",
        "paged": True,
        "limit": 100,
        "max_pages": 20,
        "params": lambda d: {
            "domain": d,
            "country_id": COUNTRY_ID,
            "page": 1,
            "limit": 100,
        },
    },
    {
        "name": "get_backlinks_referring_domains",
        "spec_path": "/api/backlinks/reports/references/getDomains",
        "method": "post",
        "desc": "Referring domains (backlinks)",
        "paged": True,  # Sprawdzi response - mo|e mie paginacj
        "limit": 100,
        "max_pages": 50,
        "params": lambda d: {
            "domain": d,
            "page": 1,
            "limit": 100,
        },
    },
    {
        "name": "get_backlinks_tlds",
        "spec_path": "/api/backlinks/reports/references/getTlds",
        "method": "post",
        "desc": "RozkBad TLD linków (backlinks)",
        "paged": False,  # Zwraca agregacj, prawdopodobnie nie paged
        "params": lambda d: {
            "domain": d,
        },
    },
    {
        "name": "get_backlinks_anchor_cloud",
        "spec_path": "/api/backlinks/reports/anchors/getCloud",
        "method": "post",
        "desc": "Anchor text cloud (backlinks)",
        "paged": False,  # Zwraca cloud data
        "params": lambda d: {
            "domain": d,
            "competitors": [],
        },
    },
    {
        "name": "get_backlinks_anchor_data",
        "spec_path": "/api/backlinks/reports/anchors/getData",
        "method": "post",
        "desc": "Anchor text data (backlinks)",
        "paged": True,  # Sprawdzi response
        "limit": 100,
        "max_pages": 30,
        "params": lambda d: {
            "domain": d,
            "competitors": [],
            "page": 1,
            "limit": 100,
        },
    },
    {
        "name": "get_competitors_analysis",
        "spec_path": "/api/visibility_analysis/tools/competitors_analysis/getData",
        "method": "post",
        "desc": "HC vs competitor analysis (common keywords)",
        "paged": False,  # Sprawdzi response - mo|e by paged
        "special_case": "hc_vs_all",  # Marker dla specjalnej logiki
        "params": lambda d: {
            "main_domain": {
                "domain": "horyzontcapital.com",
                "gte": 1,
                "lte": 50,
            },
            "country_id": COUNTRY_ID,
            "mode": "common_keywords",
            "competitors_domains": [
                {
                    "domain": d,  # competitor domain
                    "gte": 1,
                    "lte": 50,
                }
            ],
        },
    },
]


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------


def clean_domain(value: str) -> str:
	d = value.strip().replace("http://", "").replace("https://", "")
	if d.startswith("www."):
		d = d[4:]
	return d.rstrip("/")


def stem_domain(value: str) -> str:
	d = clean_domain(value)
	suffixes = [
		".com.pl",
		".net.pl",
		".org.pl",
		".info.pl",
		".pl",
		".com",
		".eu",
		".org",
		".info",
		".net",
	]
	for s in suffixes:
		if d.endswith(s):
			d = d[: -len(s)]
			break
	parts = d.split(".")
	return max(parts, key=len) if parts else d


def headers() -> Dict[str, str]:
	return {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}


def ensure_dirs() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	PARTIAL_DIR.mkdir(parents=True, exist_ok=True)


def load_spec() -> Dict[str, Any]:
	with open(SPEC_PATH, "r", encoding="utf-8") as f:
		return json.load(f)


def build_endpoints(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
	paths = spec.get("paths", {})
	endpoints: List[Dict[str, Any]] = []
	for cfg in ENDPOINT_CONFIGS:
		path = cfg["spec_path"]
		method = cfg.get("method", "get")
		spec_node = paths.get(path, {})
		spec_method = spec_node.get(method) if isinstance(spec_node, dict) else None
		endpoints.append({
			**cfg,
			"path": path,
			"method": method,
			"in_spec": bool(spec_method),
			"summary": spec_method.get("summary") if spec_method else None,
		})
	return endpoints


def load_domains(limit: Optional[int] = None) -> List[str]:
	with open(DOMAINS_PATH, "r", encoding="utf-8") as f:
		raw = [line.strip() for line in f if line.strip()]
	cleaned: List[str] = []
	seen = set()
	for d in raw:
		c = clean_domain(d)
		if c and c not in seen:
			seen.add(c)
			cleaned.append(c)
		if limit and len(cleaned) >= limit:
			break
	return cleaned


def call(url: str, params: Dict[str, Any], method: str = "get", retry: int = 0) -> Dict[str, Any]:
	try:
		m = method.lower()
		if m == "get":
			resp = requests.get(url, headers=headers(), params=params, timeout=90)
		else:
			resp = requests.post(url, headers=headers(), json=params, timeout=90)
		status = resp.status_code
		print(f"      -> {status} {resp.url[:160]}")
		if status == 200:
			try:
				return {"ok": True, "data": resp.json(), "status": status}
			except json.JSONDecodeError as exc:
				return {"ok": False, "error": f"JSON decode: {exc}"}
		if status == 429:
			if retry < 5:
				wait = min(10 * (2 ** retry), 120)
				print(f"      RATE LIMIT {status}, wait {wait}s (retry {retry+1}/5)")
				time.sleep(wait)
				return call(url, params, method, retry + 1)
			return {"ok": False, "error": "Rate limited", "status": status}
		if status == 401:
			return {"ok": False, "error": "HTTP 401", "status": status}
		return {"ok": False, "error": f"HTTP {status}: {resp.text[:200]}", "status": status}
	except requests.exceptions.Timeout:
		if retry < 2:
			time.sleep(5)
			return call(url, params, method, retry + 1)
		return {"ok": False, "error": "Timeout"}
	except requests.exceptions.RequestException as exc:
		if retry < 2:
			time.sleep(10)
			return call(url, params, method, retry + 1)
		return {"ok": False, "error": str(exc)}


def paginate(url: str, base_params: Dict[str, Any], max_pages: int, limit: int, method: str) -> Dict[str, Any]:
	items: List[Any] = []
	page = 1
	total_pages = 1
	while page <= min(total_pages, max_pages):
		params = {**base_params, "page": page, "limit": limit}
		print(f"      [PAGE {page}/{total_pages}] Fetching...")
		result = call(url, params, method)
		if not result.get("ok"):
			if page == 1:
				return result
			break

		# DEBUG: Po linii 334 - pokazuje struktur response dla pierwszej strony
		if page == 1:
			print(f"      [DEBUG] Response structure:")
			data_debug = result.get("data", {})
			if isinstance(data_debug, dict):
				print(f"        Keys in 'data': {list(data_debug.keys())}")
				if "pagination" in data_debug:
					print(f"        pagination: {data_debug['pagination']}")
				if "meta" in data_debug:
					print(f"        meta: {data_debug['meta']}")
			elif isinstance(data_debug, list):
				print(f"        'data' is list with {len(data_debug)} items")

		data = result.get("data", {})
		page_items: List[Any] = []
		if isinstance(data, list):
			page_items = data
		elif isinstance(data, dict):
			for key in [
				"data",
				"items",
				"results",
				"keywords",
				"competitors",
				"urls",
				"subdomains",
				"all_urls",
				"all_subdomains",
			]:
				val = data.get(key)
				if isinstance(val, list):
					page_items = val
					break
			pagination = data.get("pagination", data.get("meta", {}))
			if isinstance(pagination, dict):
				old_total = total_pages
				total_pages = pagination.get(
					"page_count",
					pagination.get("total_pages", pagination.get("last_page", total_pages)),
				) or total_pages
				if page == 1 and old_total != total_pages:
					print(f"      [DEBUG] Found pagination: total_pages updated from {old_total} to {total_pages}")

		items.extend(page_items)
		print(f"      [PAGE {page}] Collected {len(page_items)} items (total: {len(items)})")

		if not page_items:
			print(f"      [DEBUG] No items on page {page}, stopping pagination")
			break
		page += 1
		if page <= min(total_pages, max_pages):
			time.sleep(DELAY)
		else:
			print(f"      [DEBUG] Reached limit: page={page}, total_pages={total_pages}, max_pages={max_pages}")
	return {"ok": True, "data": items, "total": len(items)}


def fetch_endpoint(ep: Dict[str, Any], domain: str) -> Dict[str, Any]:
	url = f"{API_BASE}{ep['path']}"
	base_params = ep["params"](domain)
	method = ep.get("method", "get")

	if ep.get("multi_char"):
		data: Dict[str, Any] = {}
		ok_count = 0
		for ch in CHARACTERISTICS:
			p = {**base_params, "characteristics": ch}
			print(f"    char={ch}")
			result = call(url, p, method)
			if result.get("ok"):
				ok_count += 1
			data[ch] = result.get("data", result.get("error"))
			time.sleep(DELAY)
		return {
			"ok": ok_count > 0,
			"data": data,
			"chars_ok": ok_count,
			"chars_total": len(CHARACTERISTICS),
		}

	if ep.get("paged"):
		max_pages = ep.get("effective_max_pages", ep.get("max_pages", 50))
		return paginate(url, base_params, max_pages, ep.get("limit", 100), method)

	return call(url, base_params, method)


def save_partial(ep_name: str, domain: str, payload: Dict[str, Any]) -> None:
	safe = domain.replace("/", "_").replace(".", "_")
	ts = int(time.time())
	fp = PARTIAL_DIR / f"{ep_name}__{safe}__{ts}.json"
	try:
		with open(fp, "w", encoding="utf-8") as f:
			json.dump({"domain": domain, "endpoint": ep_name, "data": payload}, f, indent=2, ensure_ascii=False)
	except OSError:
		pass


def save_results(ep: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
	out_path = OUTPUT_DIR / f"{ep['name']}.json"
	ok = sum(1 for r in records if r.get("ok"))
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"endpoint": ep["name"],
				"path": ep["path"],
				"desc": ep.get("desc"),
				"ok": ok,
				"total": len(records),
				"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
				"country_id": COUNTRY_ID,
				"fetch_mode": FETCH_MODE,
				"results": records,
			},
			f,
			indent=2,
			ensure_ascii=False,
		)
	try:
		rows: List[Dict[str, Any]] = []
		for rec in records:
			if not rec.get("ok"):
				continue
			data = rec.get("data")
			if not isinstance(data, dict):
				continue
			flat = {"domain": rec.get("domain", "")}
			for k, v in data.items():
				if isinstance(v, (str, int, float, bool)) or v is None:
					flat[k] = v
				elif isinstance(v, dict):
					for k2, v2 in v.items():
						if isinstance(v2, (str, int, float, bool)) or v2 is None:
							flat[f"{k}__{k2}"] = v2
			if len(flat) > 1:
				rows.append(flat)
		if rows:
			csv_path = out_path.with_suffix(".csv")
			keys: List[str] = []
			for r in rows:
				for k in r:
					if k not in keys:
						keys.append(k)
			with open(csv_path, "w", encoding="utf-8", newline="") as f:
				writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
				writer.writeheader()
				writer.writerows(rows)
	except Exception:
		pass


def load_existing(ep: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
	out_path = OUTPUT_DIR / f"{ep['name']}.json"
	if not out_path.exists():
		return {}
	try:
		with open(out_path, "r", encoding="utf-8") as f:
			payload = json.load(f)
		done: Dict[str, Dict[str, Any]] = {}
		for rec in payload.get("results", []):
			if rec.get("ok") and rec.get("domain"):
				done[rec["domain"]] = rec
		if done:
			print(f"  Resume: {len(done)} domains from {out_path.name}")
		return done
	except Exception:
		return {}


def validate_token() -> bool:
	url = f"{API_BASE}/api/visibility_analysis/reports/dashboard/getDomainStatistics"
	test_domain = "horyzontcapital.com"

	print(f"\n  [1/3] Testing token with country_id={COUNTRY_ID}...")
	result = call(url, {
		"domain": test_domain,
		"fetch_mode": "topLevelDomain",
		"country_id": COUNTRY_ID
	}, "get")

	if not result.get("ok"):
		if result.get("status") == 401:
			print("  ERROR: token expired (401)")
			return False
		print(f"  WARN: token check failed: {result.get('error')}")
		return True

	print(f"  Token OK with country_id={COUNTRY_ID}")
	data_200 = result.get("data", {})

	# Test country_id=1 (Poland)
	print(f"\n  [2/3] Testing with country_id=1 (Poland)...")
	result_pl = call(url, {
		"domain": test_domain,
		"fetch_mode": "topLevelDomain",
		"country_id": "1"
	}, "get")

	if result_pl.get("ok"):
		data_1 = result_pl.get("data", {})
		print(f"  SUCCESS with country_id=1")

		# Compare
		if isinstance(data_200, dict) and isinstance(data_1, dict):
			vis_200 = data_200.get("visibility", {}).get("current") if "visibility" in data_200 else None
			vis_1 = data_1.get("visibility", {}).get("current") if "visibility" in data_1 else None
			top50_200 = data_200.get("positions", {}).get("TOP50") if "positions" in data_200 else None
			top50_1 = data_1.get("positions", {}).get("TOP50") if "positions" in data_1 else None

			print(f"\n  [3/3] COMPARISON for {test_domain}:")
			print(f"    country_id=200: visibility={vis_200}, TOP50={top50_200}")
			print(f"    country_id=1:   visibility={vis_1}, TOP50={top50_1}")

			if vis_200 == vis_1 and top50_200 == top50_1:
				print(f"\n   Results IDENTICAL - country_id 200 and 1 are equivalent")
			else:
				print(f"\n    Results DIFFERENT - consider changing COUNTRY_ID to 1")
	else:
		print(f"  FAILED with country_id=1: {result_pl.get('error')}")

	return True


def generate_call_plan(endpoints: List[Dict[str, Any]], domains: List[str]) -> List[Dict[str, Any]]:
	plan: List[Dict[str, Any]] = []
	for ep in endpoints:
		for d in domains:
			base = ep["params"](d)
			if ep.get("multi_char"):
				for ch in CHARACTERISTICS:
					p = {**base, "characteristics": ch}
					plan.append({"endpoint": ep["name"], "domain": d, "path": ep["path"], "method": ep.get("method", "get"), "params": p})
			else:
				plan.append({"endpoint": ep["name"], "domain": d, "path": ep["path"], "method": ep.get("method", "get"), "params": base})
	plan_path = OUTPUT_DIR / "_call_plan.json"
	with open(plan_path, "w", encoding="utf-8") as f:
		json.dump({"total": len(plan), "plan": plan}, f, indent=2, ensure_ascii=False)
	print(f"  Call plan saved: {plan_path}")
	return plan


def fetch_all(ep: Dict[str, Any], domains: Iterable[str], resume: bool) -> List[Dict[str, Any]]:
	print(f"\n{'=' * 80}")
	print(f"  {ep['name']} - {ep.get('desc', '')}")
	print(f"{'=' * 80}")
	existing = load_existing(ep) if resume else {}
	results: List[Dict[str, Any]] = []
	skipped = 0
	for idx, domain in enumerate(domains, 1):
		if domain in existing:
			results.append(existing[domain])
			skipped += 1
			continue
		print(f"  [{idx}/{len(domains)}] {domain}")

		# Apply group-based pagination limits
		ep_for_domain = {**ep}
		group = get_domain_group(domain)

		if ep["name"] in GROUP_C_LIMITED_ENDPOINTS and group == "C":
			old_max = ep_for_domain.get("effective_max_pages", ep_for_domain.get("max_pages", 50))
			ep_for_domain["effective_max_pages"] = GROUP_C_MAX_PAGES
			print(f"    [GROUP C] Limiting to {GROUP_C_MAX_PAGES} pages (default: {old_max})")
		else:
			print(f"    [GROUP {group}] No page limit")

		data = fetch_endpoint(ep_for_domain, domain)
		data["domain"] = domain
		data["endpoint"] = ep["name"]
		results.append(data)
		save_partial(ep["name"], domain, data)
		if idx < len(domains):
			time.sleep(DELAY)
		if idx % 10 == 0:
			save_results(ep, results)
	if skipped:
		print(f"  Skipped {skipped} (resumed)")
	save_results(ep, results)
	ok = sum(1 for r in results if r.get("ok"))
	print(f"  DONE: {ok}/{len(results)}")
	return results


def run(args: argparse.Namespace) -> None:
	ensure_dirs()
	spec = load_spec()
	endpoints = build_endpoints(spec)
	domains = load_domains(args.first if args.first else None)

	print("=" * 80)
	print("  SENUTO API - RUNDA 1")
	print(f"  Domains: {len(domains)} | Endpoints: {len(endpoints)} | Delay: {DELAY}s")
	print("=" * 80)

	missing = [e for e in endpoints if not e.get("in_spec")]
	if missing:
		print("  WARN: endpoints not found in spec:")
		for m in missing:
			print(f"    - {m['name']} ({m['path']})")

	if not args.skip_validate and not validate_token():
		print("  Aborting due to token check failure")
		return

	if args.plan_only:
		generate_call_plan(endpoints, domains)
		return

	generate_call_plan(endpoints, domains)

	all_results: Dict[str, Any] = {}
	start = time.time()
	for idx, ep in enumerate(endpoints, 1):
		# shallow copy so we can adjust pagination per run
		ep_run = {**ep}
		if ep_run.get("paged"):
			if args.first_page_only:
				ep_run["effective_max_pages"] = 1
			else:
				ep_run["effective_max_pages"] = ep_run.get("max_pages", 50)
		print(f"\n{'#' * 80}")
		print(f"  ENDPOINT {idx}/{len(endpoints)}: {ep_run['name']}")
		print(f"{'#' * 80}")
		try:
			all_results[ep_run["name"]] = fetch_all(ep_run, domains, resume=not args.no_resume)
		except KeyboardInterrupt:
			print("\n  Interrupted by user")
			raise
		except Exception as exc:
			print(f"  ERROR {ep_run['name']}: {exc}")
			all_results[ep_run["name"]] = [{"error": str(exc)}]

	elapsed = time.time() - start
	summary = {
		"completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"elapsed_min": round(elapsed / 60, 1),
		"domains": len(domains),
		"endpoints": {
			name: {
				"ok": sum(1 for r in res if isinstance(r, dict) and r.get("ok")),
				"total": len(res),
			}
			for name, res in all_results.items()
			if isinstance(res, list)
		},
	}
	with open(OUTPUT_DIR / "_summary_runda1.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)
	print(f"\n  Summary saved to {OUTPUT_DIR / '_summary_runda1.json'}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Senuto Round 1 executor")
	parser.add_argument("--first", type=int, default=0, help="Process only the first N domains")
	parser.add_argument("--plan-only", action="store_true", help="Only build call plan, no requests")
	parser.add_argument("--no-resume", action="store_true", help="Ignore previous results when running")
	parser.add_argument("--skip-validate", action="store_true", help="Skip token validation call")
	parser.add_argument("--first-page-only", action="store_true", help="Fetch only first page for testing (default: fetch all pages up to max_pages)")
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
	args = parse_args(argv)
	try:
		run(args)
	except KeyboardInterrupt:
		print("\nInterrupted  partial data saved in partial/")
		sys.exit(130)


if __name__ == "__main__":
	main()
