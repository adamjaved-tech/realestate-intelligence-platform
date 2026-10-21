terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "me-central1"
}
variable "image" {
  type        = string
  description = "Container image for the API, e.g. REGION-docker.pkg.dev/PROJECT/propintel/api:TAG"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Data lake (bronze/silver/gold parquet) + model artifacts
resource "google_storage_bucket" "lake" {
  name                        = "${var.project_id}-propintel-lake"
  location                    = var.region
  uniform_bucket_level_access = true
  versioning { enabled = true }
  lifecycle_rule {
    condition { age = 90 }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "propintel"
  format        = "DOCKER"
  location      = var.region
}

resource "google_service_account" "api" {
  account_id   = "propintel-api"
  display_name = "PropIntel API runtime"
}

# least privilege: read-only access to the lake
resource "google_storage_bucket_iam_member" "api_read" {
  bucket = google_storage_bucket.lake.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name     = "propintel-api"
  location = var.region
  template {
    service_account = google_service_account.api.email
    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }
    containers {
      image = var.image
      ports { container_port = 8000 }
      resources {
        limits = { cpu = "2", memory = "2Gi" }
      }
      startup_probe {
        http_get { path = "/health" }
        period_seconds = 5
      }
    }
  }
}

output "api_url" { value = google_cloud_run_v2_service.api.uri }
output "lake_bucket" { value = google_storage_bucket.lake.name }
