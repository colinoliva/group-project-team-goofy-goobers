# Project Title

Texas Instruments Raspberry Pi mmWave Radar Sensor 

## Team

| Name | GitHub | Email |
|------|--------|-------|
| Paul Brandon Estigoy | [@Brannflakess](https://github.com/Brannflakess) | paulbrandon.estigoy@sjsu.edu |
| Raghav Gautam| [@RaghavG17](https://github.com/RaghavG17) | raghav.gautam@sjsu.edu |
| Colin Oliva | [@colinoliva](https://github.com/colinoliva) | colin.oliva@sjsu.edu |
| Ernest Ma | [@maernest04](https://github.com/maernest04) | ernest.ma@sjsu.edu |

**Advisor:** Charan Bhaskar

---

## Problem Statement

Achieving reliable and affordable 3D environment mapping remains a significant hurdle for DIY developers, robotics teams, and small businesses. While industrial-grade LiDAR offers high precision, its prohibitive cost limits its accessibility. Conversely, low-cost radar sensors frequently suffer from signal noise and false positives when operated as standalone units, leading to untrustworthy spatial data in consumer applications like surveillance and monitoring. The core technical challenge lies in the inherent limitations of single-sensor deployments; however, by synchronizing multiple sensors in a unified array, it is possible to mitigate noise and achieve high-fidelity 3D mapping that rivals expensive industrial systems at a fraction of the cost.


## Solution

Our solution leverages an array of synchronized Texas Instruments mmWave radar sensors interfaced with a Raspberry Pi to provide high-precision 3D mapping at a consumer-friendly price point. By implementing an edge-based processing pipeline, the system utilizes advanced spatial algorithms—including DBSCAN clustering, trilateration, and voxel downsampling—to aggregate data from multiple vantage points and filter out signal noise. This synchronized multi-sensor approach drastically reduces false positives and provides a robust, real-time 3D visualization, offering a reliable alternative to expensive industrial LiDAR systems.

### Key Features

- Feature 1
- Feature 2
- Feature 3

---

## Demo

[Link to demo video or GIF]

**Live Demo:** [URL if deployed]

---

## Screenshots

| Feature | Screenshot |
|---------|------------|
| [Feature 1] | ![Screenshot](docs/screenshots/feature1.png) |
| [Feature 2] | ![Screenshot](docs/screenshots/feature2.png) |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | |
| Backend | |
| Database | |
| Deployment | |

---

## Getting Started

### Prerequisites

- [Prerequisite 1] v.X.X+
- [Prerequisite 2] v.X.X+

### Installation

```bash
# Clone the repository
git clone https://github.com/[org]/[repo].git
cd [repo]

# Install dependencies
[install command]

# Set up environment variables
cp .env.example .env
# Edit .env with your values

# Run database migrations (if applicable)
[migration command]
```

### Running Locally

```bash
# Development mode
[dev command]

# The app will be available at http://localhost:XXXX
```

### Running Tests

```bash
[test command]
```

---

## API Reference

<details>
<summary>Click to expand API endpoints</summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/resource` | Get all resources |
| GET | `/api/resource/:id` | Get resource by ID |
| POST | `/api/resource` | Create new resource |
| PUT | `/api/resource/:id` | Update resource |
| DELETE | `/api/resource/:id` | Delete resource |

</details>

---

## Project Structure

```
.
├── [folder]/           # Description
├── src/                # Source code files
├── tests/              # Test files
├── docs/               # Documentation files
└── README.md
```

---

## Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:
- `Add user authentication endpoint`
- `Fix database connection timeout issue`
- `Update README with setup instructions`

---

## Acknowledgments

- [Resource/Library/Person]
- [Resource/Library/Person]

---

## License

This project is licensed under the <FILL IN> License - see the [LICENSE](LICENSE) file for details.

---

*CMPE 195A/B - Senior Design Project | San Jose State University | Spring 2026*
