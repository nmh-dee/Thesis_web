from django.urls import path, reverse_lazy
from . import views

app_name = 'abns'
urlpatterns = [
    path('', views.AbnListView.as_view(), name = 'all'),
    path('abn/<int:pk>', views.AbnDetailView.as_view(), name = 'abn_detail'),
    path('abn/create', views.AbnCreateView.as_view(), name= 'abn_create'),
    path('abn/<int:pk>/update', views.AbnUpdateView.as_view(), name= 'abn_update'),
    path('abn/<int:pk>/delete', views.AbnDeleteView.as_view(success_url = reverse_lazy('abns:all')), name = 'abn_delete')
]