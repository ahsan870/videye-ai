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

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/videye-ai.git
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
   - Upload a video file (MP4 or AVI format)
   - Select the road type (4-way intersection, T-junction, or 2-way road)
   - Choose the camera position
   - Click "Process Video"
   - View the analysis results

## Analysis Features

### Movement Analysis
- Vehicle count
- Direction distribution
- Turn patterns (left, right, straight)
- Stationary vehicle detection

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
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please open an issue in the GitHub repository or contact [your-email@example.com].

## Acknowledgments
- YOLO for object detection
- Streamlit for the web framework
- OpenCV for video processing capabilities