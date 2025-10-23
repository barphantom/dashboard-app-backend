from django.contrib.auth import authenticate
from rest_framework import status, generics
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from users.serializers import RegisterSerializer


class RegisterView(generics.CreateAPIView):
    permission_classes = [AllowAny]
    serializer_class = RegisterSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response({
            "user": {
                "email": user.email,
                "name": user.name,
                "lastName": user.lastName,
            },
            "access": str(refresh.access_token),
            "refresh": str(refresh),
        }, status=status.HTTP_201_CREATED)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(request, email=email, password=password)

        if not user:
            return Response({
                "error": "Invalid credentials",
            }, status=status.HTTP_401_UNAUTHORIZED)

        refresh = RefreshToken.for_user(user)
        return Response({
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "user": {
                "email": user.email,
                "name": user.name,
                "lastName": user.lastName,
            }
        })


        # email = request.data.get('email')
        # password = request.data.get('password')
        # if not email or not password:
        #     return Response({'detail': "Email and password are required"}, status=status.HTTP_400_BAD_REQUEST)
        #
        # try:
        #     user_obj = User.objects.get(email=email)
        # except User.DoesNotExist:
        #     return Response({'detail': "Wrong login data"}, status=status.HTTP_401_UNAUTHORIZED)
        #
        # user = authenticate(request, username=user_obj.username, password=password)
        # if user is None:
        #     return Response({'detail': "Wrong login data"}, status=status.HTTP_401_UNAUTHORIZED)
        #
        # refresh = RefreshToken.for_user(user)
        # return Response({
        #     'user': UserSerializer(user).data,
        #     'access': str(refresh.access_token),
        #     'refresh': str(refresh),
        # })

# class MeView(APIView):
#     permission_classes = [IsAuthenticated]
#
#     def get(self, request):
#         serializer = UserSerializer(request.user)
#         return Response(serializer.data)
