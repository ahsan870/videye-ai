# VidEye AI - Traffic Intersection Monitoring System

## Overview
VidEye AI is an advanced traffic monitoring system that uses artificial intelligence to analyze vehicle movements at different types of road intersections. The application processes video footage to track vehicles and provide detailed analytics about traffic patterns and movement distributions.

## Features
- 🎥 Real-time video processing
- 🚗 Vehicle detection and tracking
- 📊 Comprehensive traffic analytics
- 🛣️ Support for multiple road types:
  - 4-way intersections
  - T-junctions
  - 2-way roads
- 📍 Multiple camera position support
- 📈 Detailed movement analysis
- 📥 Exportable results in CSV format

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam or video file for analysis
- Modern web browser (Chrome, Firefox, Safari)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/ahsan870/videye-ai.git
cd videye-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Follow these steps to analyze traffic:
   - Upload a video file (MP4 or AVI format, max size 200MB)
   - Select the road type (4-way intersection, T-junction, or 2-way road)
   - Choose the appropriate camera position based on your video perspective
   - Click "Process Video" to start analysis
   - Use Pause/Resume controls during processing if needed
   - View and download the analysis results

## Analysis Features

### Movement Analysis
- Real-time vehicle counting
- Directional flow analysis
- Turn pattern detection (left, right, straight)
- Stationary vehicle identification
- Traffic density metrics

### Visualization
- Interactive charts
- Movement distribution graphs
- Detailed metrics
- Exportable reports

## Technical Details

### Built With
- Streamlit - Web interface
- OpenCV - Video processing
- YOLOv8 - Object detection
- PyTorch - Deep learning framework
- Pandas - Data analysis

### System Requirements
- RAM: 8GB minimum (16GB recommended)
- Storage: 1GB free space
- GPU: Optional but recommended for faster processing

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
Proprietary Software - All Rights Reserved

Copyright (c) 2024 VidEye AI

This software and associated documentation files (the "Software") are proprietary and confidential. 
All rights are reserved. No part of this Software may be reproduced, distributed, or transmitted in any form 
or by any means, including photocopying, recording, or other electronic or mechanical methods, without 
the prior written permission of the copyright holder.

### Commercial Usage
This software can be used for commercial purposes under the following conditions:
1. Purchase of appropriate commercial license
2. Written agreement with the copyright holder
3. Compliance with terms of service and usage guidelines
4. Adherence to data privacy regulations

### Prohibited Actions
- Redistribution of the source code
- Modification of the source code without permission
- Resale without authorization
- Removal of copyright notices
- Use in competing products
- Reverse engineering of the software

For commercial licensing and pricing inquiries, please contact:
Email: ahmed.bitsandbytes@gmail.com
Website: 

## Support
For technical support and inquiries:
- Email: ahmed.bitsandbytes@gmail.com
- GitHub Issues: 
- Documentation: 

Response time: Within 24-48 business hours

## Acknowledgments
- YOLO for object detection technology
- Streamlit for the web application framework
- OpenCV for video processing capabilities
- Our beta testers and early adopters