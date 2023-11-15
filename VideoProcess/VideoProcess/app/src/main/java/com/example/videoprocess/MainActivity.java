package com.example.videoprocess;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.widget.MediaController;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import android.widget.VideoView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;


public class MainActivity extends AppCompatActivity {
    private Uri videoUri;
    private static final int REQUEST_VIDEO_CAPTURE = 1;
    // Create an ActivityResultLauncher for handling video selection

    private static final int REQUEST_STORAGE_PERMISSIONS = 100;

    private void requestStoragePermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_STORAGE_PERMISSIONS);
        } else {
            // Permissions are already granted, continue with your task
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_STORAGE_PERMISSIONS) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permissions are granted, continue with your task
            } else {
                Toast.makeText(this, "Storage permissions are required to access the file.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private final ActivityResultLauncher<Intent> videoResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    videoUri = result.getData().getData();
                    uploadVideoToServer();
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        requestStoragePermissions();
        Button selectVideoButton = findViewById(R.id.select_video_button);
        selectVideoButton.setOnClickListener(this::selectVideo);
    }
    public void switchActivity(View view) {
        Intent intent = new Intent(this, home_activity.class);
        startActivity(intent);
    }

    public void selectVideo(View view) {
        Intent intent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        videoResultLauncher.launch(intent);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_VIDEO_CAPTURE && resultCode == RESULT_OK) {
            videoUri = data.getData();

            uploadVideoToServer();
        }
    }

    private void uploadVideoToServer() {
        if (videoUri == null) {
            Toast.makeText(this, "Please select a video", Toast.LENGTH_SHORT).show();
            return;
        }

        File videoFile = new File(PathUtil.getPath(this, videoUri));
        MediaType mediaType = MediaType.parse(getContentResolver().getType(videoUri));
        RequestBody videoRequestBody = RequestBody.create(MediaType.parse("video/mp4"), videoFile);
        MultipartBody.Part videoPart = MultipartBody.Part.createFormData("file", videoFile.getName(), videoRequestBody);

        ApiService apiService = RetrofitClient.getClient("http://10.0.2.2:8000").create(ApiService.class);
        Call<ResponseBody> call = apiService.uploadVideo(videoPart);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                if (response.isSuccessful()) {
                    Toast.makeText(MainActivity.this, "Video uploaded successfully", Toast.LENGTH_SHORT).show();
                    String outputFilePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + "/processed_video.mp4";
                    try (InputStream inputStream = response.body().byteStream();
                         OutputStream outputStream = new FileOutputStream(outputFilePath)) {

                        byte[] buffer = new byte[4096];
                        int bytesRead;

                        while ((bytesRead = inputStream.read(buffer)) != -1) {
                            outputStream.write(buffer, 0, bytesRead);
                        }

                        Log.d("VideoProcessing", "Processed video saved to " + outputFilePath);

                        //set selcted video
                        VideoView videoView2= findViewById(R.id.videoView2);
                        videoView2.setVideoURI(videoUri);
                        videoView2.start();
                        // Find the VideoView in the layout
                        VideoView videoView = findViewById(R.id.videoView);

                        // Set the video file URI
                        Uri videoUri = Uri.parse(outputFilePath);
                        videoView.setVideoURI(videoUri);
                        //set media controller
                        MediaController mediaController = new MediaController(MainActivity.this);
                        mediaController.setAnchorView(videoView);
                        videoView.setMediaController(mediaController);
                        // Start playing the video
                        videoView.start();
                    } catch (IOException e) {
                        Log.e("VideoProcessing", "Error saving processed video", e);
                    }
                } else {
                    Toast.makeText(MainActivity.this, "Upload failed", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Toast.makeText(MainActivity.this, "Error: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                Log.d("Error",t.getMessage());
            }

        });
    }
}