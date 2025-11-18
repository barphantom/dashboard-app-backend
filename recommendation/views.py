from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from recommendation.serializers import RecommendationRequestSerializer, GARecommendationRequestSerializer
from recommendation.utils_cosine import get_recommendations
from recommendation.utils_genetic import get_ga_recommendations


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def recommendation_view(request):
    serializer = RecommendationRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    sector_data = data['sector_allocations']
    global_params = data["global_params"]

    if not sector_data or not global_params:
        return Response({"error": "Missing data"}, status=400)

    try:
        recommendations = get_recommendations(sector_data, global_params)
        return Response({"recommendations": recommendations}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"Error generatin recommendations": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def ga_recommendation_view(request):
    """
    Endpoint dla algorytmu genetycznego:
    POST /api/recommendations/ga/
    """
    serializer = GARecommendationRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    params = serializer.validated_data

    try:
        result = get_ga_recommendations(params)
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": "Error generating GA recommendations", "details": str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)
