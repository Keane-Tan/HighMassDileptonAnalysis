# Datasets
## 2022preEE
### Data
| Sample    | DAS |
| --------- | --- |
| B | /SingleMuon/Run2022B-22Sep2023-v1/NANOAOD
| C | /Muon/Run2022C-22Sep2023-v1/NANOAOD
|   | /SingleMuon/Run2022C-22Sep2023-v1/NANOAOD
| D | /Muon/Run2022D-22Sep2023-v1/NANOAOD

I'm not using Egamma samples for now, because I haven't updated the trigger logic in our code to avoid double counting events. I will do it soon.

### Background MC
| Sample Group | Sample | DAS |
| ------------ | ------ | --- |
| DY | dyTo2L_M-4_incl | /DYto2L-2Jets_MLL-4to10_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
| DY | dyTo2L_M-10_incl | /DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v1/NANOAODSIM
| DY | dyTo2L_M-50_incl | /DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v1/NANOAODSIM
| Top | tt_inclusive | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM
| VV | ww_2l2nu | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM
| VV | ww_4q | /WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM
| VV | ww_lnu2q | /WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM
| VV | wz | /WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
| VV | wz_3lnu | /WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
| VV | wz_2l2q | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM
| VV | wz_lnu2q | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
|   |    | /WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM
| VVV | zzz | /ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
| VVV | wzz | /WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
| VVV | wwz_4f | /WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
| VVV | www_4f | /WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM

When making the plots, I did not include `w_lnu` and `wz_4q` samples because I don't have their cross sections.

# Cross sections
| Samples          | Cross section (pb) |
| ---------------- | ------------------ |
| dyTo2L_M-4_incl  | 165700             |
| dyTo2L_M-10_incl | 28080              |
| dyTo2L_M-50_incl | 11640              |
| tt_inclusive     | 88.51              |
| ww_2l2nu         | 12.178             |
| ww_4q            | 51.723             |
| ww_lnu2q         | 49.997             |
| wz               | 47.13              |
| wz_3lnu          | 4.924              |
| wz_2l2q          | 5.595              |
| wz_lnu2q         | 10.71              |
| zzz              | 0.01398            |
| wzz              | 0.05565            |
| wwz_4f           | 0.1651             |
| www_4f           | 0.2086             |

Took the cross section values from https://twiki.cern.ch/twiki/bin/view/CMS/SummaryTable1G25ns, except for the DY samples. Cross sections for the DY samples came from `d_xSec_all_NEW["22"]` in https://gitlab.cern.ch/rocavigl/dileptonsAtRun3/-/blob/master/tools/helper_scripts/physics.py

# Luminosity
2022PreEE: 7.9804 fb-1

Taken from `d_lumi['22']` in https://gitlab.cern.ch/rocavigl/dileptonsAtRun3/-/blob/master/tools/helper_scripts/physics.py

# Muon selections
| Variable | Cut |
| -------- | --- |
| pT | > 25 |
| abs(eta) | < 2.5 |
| highPtId | == 2 |
| abs(dxy) | < 0.5 |
| abs(dz) | < 0.1 |
| tkRelIso | < 0.1 |

# Triggers (OR Requirement; accept event as long as it passes any of the triggers below)
* IsoMu24
* IsoMu24
* Mu50
* HighPtTkMu100
* CascadeMu100

`HLT_Mu50_L1SingleMuShower` trigger doesn't exist in NanoAODv12 sample (see https://cms-nanoaod-integration.web.cern.ch/autoDoc/). There is however `L1_SingleMuShower_Nominal`, but that's not an HLT.
