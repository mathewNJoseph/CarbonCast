#!/bin/sh
# Copy the tier 2 inputs of k8s/concordance-regions.yaml onto spatial-rook, through the cc-upload pod from
# k8s/upload-concordance-inputs.yaml, as spatial-rook:/carboncast/inputs/<R>/<R>_{168hr_forecasts_DA,direct_emissions}.csv
#
#   sh k8s/upload_concordance_inputs.sh ../CarbonCast-SARIMAX-mat AZPS BPAT CISO ...
#
# The DA files come from <SARIMAX repo>/s3_tar_files and the direct emissions from <SARIMAX repo>/direct_emissions_copies.
set -eu
[ $# -ge 2 ] || { echo "usage: $0 SARIMAX_REPO REGION [REGION ...]" >&2; exit 2; }
SRC=$1; shift
NS=cse239fall2025
POD=cc-upload

for R in "$@"; do
    da=$SRC/s3_tar_files/${R}_168hr_forecasts_DA.csv
    em=$SRC/direct_emissions_copies/${R}_direct_emissions.csv
    for f in "$da" "$em"; do
        [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
    done
    kubectl exec -n $NS $POD -- mkdir -p /spatial-rook/carboncast/inputs/$R
    kubectl cp -n $NS "$da" $POD:/spatial-rook/carboncast/inputs/$R/${R}_168hr_forecasts_DA.csv
    kubectl cp -n $NS "$em" $POD:/spatial-rook/carboncast/inputs/$R/${R}_direct_emissions.csv
    echo "$R: $(kubectl exec -n $NS $POD -- ls -l /spatial-rook/carboncast/inputs/$R | tail -n +2 | wc -l) files"
done
