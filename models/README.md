# Model Files

## Pretrained Models Not Included

Due to file size limitations, pretrained model files are **not included** in this Git repository.

### Required Models

For full functionality, you need the following pretrained models:

#### EGO Agent Models (Ego Vehicle - Trained agent)
- **Highway**: `models/EGO/highway/table_EGO_TRAIN_2000.txt`
- **Intersection**: `models/EGO/intersection/table_EGO_TRAIN_8000.txt`

#### ADV Agent Models (Adversarial - For testing)
Located in `models/ADV/{environment}/{algorithm}/`

### Options to Get Models

#### Option 1: Train Your Own (Recommended)
```bash
# Train Q-learning agent for highway environment
python scripts/train.py --config configs/highway.yaml

# Models will be saved to models/ADV/{environment}/Q/
```

#### Option 2: Download Pretrained Models
If pretrained models are available separately:
1. Download from the releases page or external storage
2. Extract to the `models/` directory
3. Ensure directory structure matches:
```
models/
├── EGO/
│   ├── highway/
│   │   └── table_EGO_TRAIN_2000.txt
│   └── intersection/
│       └── table_EGO_TRAIN_8000.txt
└── ADV/
    ├── highway/
    │   ├── Q/
    │   ├── DQN/
    │   └── CRL/
    └── intersection/
        └── Q/
```

#### Option 3: Use Without Pretrained EGO Models
You can train from scratch without pretrained EGO models:
```python
# In your training script, create a new ego agent
from carlet.agents import QTableAgent

ego_agent = QTableAgent(state_size=10, action_size=5)
# Train from scratch
```

### Model File Formats

- **Q-Table models**: `.txt` files with key-value pairs
- **DQN models**: TensorFlow `.h5` or `.weights.h5` files
- **Other models**: Depends on the algorithm

### Note for Contributors

If you train models and want to share them:
1. DO NOT commit large model files to Git
2. Use Git LFS or external storage (Google Drive, OneDrive, etc.)
3. Provide download links in releases or documentation
