import requests
import json
import pandas as pd
from typing import List, Optional

class TrialIngestor:
    def __init__(self, base_url: str = "https://clinicaltrials.gov/api/v2/studies"):
        self.base_url = base_url

    def fetch_trials(self, query: str, max_results: int = 100) -> List[dict]:
        print(f"Fetching trials for query: {query}...")
        params = {
            "query.cond": query,
            "pageSize": max_results,
            "format": "json"
        }
        
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            studies = data.get('studies', [])
            print(f"Successfully fetched {len(studies)} studies.")
            return studies
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return []

    def process_and_save(self, studies: List[dict], output_path: str):
        processed_data = []
        for study in studies:
            protocol = study.get('protocolSection', {})
            ident = protocol.get('identificationModule', {})
            design = protocol.get('designModule', {})
            status = protocol.get('statusModule', {})
            
            processed_data.append({
                'nct_id': ident.get('nctId'),
                'title': ident.get('briefTitle'),
                'phase': design.get('phases', [None])[0],
                'status': status.get('overallStatus'),
                'enrollment': design.get('enrollmentInfo', {}).get('count')
            })
        
        df = pd.DataFrame(processed_data)
        df.to_csv(output_path, index=False)
        print(f"Saved processed trials to {output_path}")

if __name__ == "__main__":
    ingestor = TrialIngestor()
    trials = ingestor.fetch_trials("Oncology", max_results=50)
    ingestor.process_and_save(trials, "data/raw/oncology_trials.csv")
