## Curl requests

### Send files for training :
curl -d "file=@data.txt" -X POST http://localhost:5000/train

### Send files for testing :
curl -d "file=@data.txt" -X POST http://localhost:5000/test

### Adding new files for training :
curl -d "file=@data.txt" -X POST http://localhost:5000/new
