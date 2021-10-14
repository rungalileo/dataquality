from enum import Enum


class Route(str, Enum):
    """
    List of available API routes
    """

    jobs = "jobs"
    projects = "projects"
    users = "users"
    cleanup = "cleanup"
