import wbdata


def get_gdp(countries):
    wb_countries = wbdata.get_country()
    countries = [c.lower() for c in countries]
    countries_id = [x["id"] for x in wb_countries if x["name"].lower() in countries]
    indicators = {"NY.GDP.PCAP.PP.KD": "gdppc"}
    return wbdata.get_dataframe(indicators, country=countries_id, convert_date=True)
