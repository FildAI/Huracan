#!/bin/bash

echo "🧹 Docker Cleanup Script"
echo "========================="

# Function to show disk usage
show_disk_usage() {
    echo "💾 Current disk usage:"
    df -h /
    echo ""
}

# Function to show Docker disk usage
show_docker_usage() {
    echo "🐳 Docker disk usage:"
    docker system df
    echo ""
}

# Show initial disk usage
echo "📊 Initial status:"
show_disk_usage
show_docker_usage

# Clean up Docker resources
echo "🗑️  Cleaning Docker resources..."

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused images
echo "Removing unused images..."
docker image prune -f

# Remove build cache
echo "Removing build cache..."
docker builder prune -f

# Remove everything unused (aggressive cleanup)
echo "Performing aggressive cleanup..."
docker system prune -a -f --volumes

echo "✅ Docker cleanup complete!"
echo ""

# Show final disk usage
echo "📊 Final status:"
show_disk_usage
show_docker_usage

echo "🎉 Cleanup complete! You can now try rebuilding your containers." 