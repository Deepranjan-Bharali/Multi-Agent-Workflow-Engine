# control_plane/workflows/admin.py
from django.contrib import admin
from .models import Project, WorkflowRunLog

admin.site.register(Project)
admin.site.register(WorkflowRunLog)