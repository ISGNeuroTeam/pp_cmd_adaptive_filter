# pp_cmd_adaptive_filter
Postprocessing command "adaptive_filter"

Adaptive filter
    
    raw_signal:  raw signal
    desired_signal: desired signal
    type: type of filter: LMS, NLMS, RLS
    mu: step of filter
    filter_size: size of filter
Usage example:
`... | adaptive_filter raw_signal desired_signal type="LMS", filter_size=32`

## Getting started
###  Prerequisites
1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installing
1. Create virtual environment with post-processing sdk 
```bash
make dev
```
That command  
- creates python virtual environment with [postprocessing_sdk](https://github.com/ISGNeuroTeam/postprocessing_sdk)
- creates `pp_cmd` directory with links to available post-processing commands
- creates `otl_v1_config.ini` with otl platform address configuration

2. Configure connection to platform in `otl_v1_config.ini`

### Test adaptive_filter
Use `pp` to test adaptive_filter command:  
```bash
pp
Storage directory is /tmp/pp_cmd_test/storage
Commmands directory is /tmp/pp_cmd_test/pp_cmd
query: | otl_v1 <# makeresults count=100 #> |  adaptive_filter 
```
